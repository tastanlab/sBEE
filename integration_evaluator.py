"""
integration_evaluator.py
------------------------
Single-cell batch integration evaluation toolkit.

Usage
-----
    from integration_evaluator import IntegrationEvaluator

    evaluator = IntegrationEvaluator(
        adata,
        label_key="cell_type",
        batch_key="batch",
        k=90,
    )
    evaluator.prepare()
    evaluator.run()
    evaluator.save(path="results/01", tag="count_k_90")
"""

import os
import numpy as np
import pandas as pd
import scib
import rpy2.robjects as ro
import anndata2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import jensenshannon, cdist

from silhouette_batch import silhouette_batch
from graph_connectivity_per_celltype import graph_connectivity_per_celltype


class IntegrationEvaluator:
    """
    Evaluates single-cell batch integration quality.

    Parameters
    ----------
    adata : AnnData
        Annotated data object (raw, before preprocessing).
    label_key : str
        Column in ``adata.obs`` containing cell-type labels.
    batch_key : str
        Column in ``adata.obs`` containing batch labels.
    k : int
        Number of nearest neighbors used throughout evaluation. Default 90.
    n_comps : int
        Number of PCA components. Default 50.
    celltypes_to_ignore : list[str] or None
        Cell types to drop before evaluation.
    local_count_min : int
        Cells whose local neighborhood has fewer than this many cells are
        excluded from per-cell-type aggregation. Default 20.
    """

    def __init__(
        self,
        adata,
        label_key: str,
        batch_key: str,
        sc_dir: str,
        k: int = 90,
        n_comps: int = 50,
        celltypes_to_ignore: list = None,
        local_count_min: int = 20,
    ):
        self.adata = adata
        self.label_key = label_key
        self.batch_key = batch_key
        self.sc_dir = sc_dir
        self.k = k
        self.n_comps = n_comps
        self.celltypes_to_ignore = celltypes_to_ignore
        self.local_count_min = local_count_min

        os.makedirs(sc_dir, exist_ok=True)

        # populated after prepare() / run()
        self.scores: pd.DataFrame = None
        self.kbet_df: pd.DataFrame = None
        self.valid_cells = None
        self._prepared = False

    # ------------------------------------------------------------------
    # Public pipeline methods
    # ------------------------------------------------------------------

    def prepare(self):
        """
        Preprocess ``adata``: rename obs columns, optionally filter cell
        types, run HVG selection, and compute PCA.
        """
        adata = self.adata

        adata.obs.rename(
            columns={self.label_key: "Group", self.batch_key: "Batch"},
            inplace=True,
        )

        if self.celltypes_to_ignore is not None:
            adata = adata[~adata.obs["Group"].isin(self.celltypes_to_ignore)].copy()

        print("Reducing data...")
        scib.preprocessing.reduce_data(adata, batch_key="Batch", pca_comps=self.n_comps)
        adata = adata[:, adata.var["highly_variable"]].copy()

        self.adata = adata
        self._prepared = True
        return self

    def run(self):
        """
        Run the full evaluation pipeline and populate ``self.scores`` and
        ``self.kbet_df``.

        Requires ``prepare()`` to have been called first.
        """
        if not self._prepared:
            raise RuntimeError("Call prepare() before run().")

        adata = self.adata
        k = self.k

        # ---- derived metadata ----------------------------------------
        celltypes_df = pd.DataFrame(adata.obs["Group"])
        batches_df = pd.DataFrame(adata.obs["Batch"])
        n_batches = batches_df["Batch"].nunique()
        pca_coords = adata.obsm["X_pca"]
        int_df = pd.DataFrame(pca_coords)

        # ---- kNN graph -----------------------------------------------
        idx, dists = self.knn(int_df, k=k)
        dknn_df = pd.DataFrame(idx, index=int_df.index)

        # ---- distributions -------------------------------------------
        local_count = self.build_distribution(
            adata, dknn_df, celltypes_df, batches_df, dist_type="count"
        )
        global_dist = self.build_distribution(
            adata, dknn_df, celltypes_df, batches_df, dist_type="global"
        )

        # ---- JS-based metrics ----------------------------------------
        scores = pd.DataFrame(0.0, columns=[], index=global_dist.index)
        scores["Cell type"] = celltypes_df
        scores["Batch"] = batches_df
        scores = self.distr_based_metrics(scores, local_count.copy(), global_dist.copy(), n_batches)

        # ---- intra/inter distances -----------------------------------
        intra_inter_df = self.compute_intra_inter_distances(adata)
        scores["intra_mean"] = intra_inter_df["intra_mean"]
        scores["inter_mean"] = intra_inter_df["inter_mean"]
        scores["intra_inter_ratio"] = intra_inter_df["intra_inter_ratio"]
        scores = self.sBEE(scores)

        # ---- valid cells (sufficient local neighborhood) -------------
        local_cutoff = local_count[local_count.sum(axis=1) > self.local_count_min]
        self.valid_cells = local_cutoff.index

        # ---- conventional metrics ------------------------------------
        ilisi_df, scib_kbet, sil_df, pcr, graph_conn = self.conventional_metrics(
            int_df, batches_df, k, adata,
            batch_key="Batch", group_key="Group", embed="X_pca",
        )
        scores["ASW"] = sil_df["silhouette_score"]

        ilisi_df["scaled"] = (ilisi_df["Batch"] - 1) / (n_batches - 1)
        scores["scaled iLISI"] = ilisi_df["scaled"]

        # ---- original kBET -------------------------------------------
        orig_kbet, orig_kbet_k = self.orig_kbet_score_k(
            adata, batch_key="Batch", group_key="Group", sc_order=self.sc_dir
        )
        kbet_df = pd.DataFrame.from_dict(orig_kbet, orient="index", columns=["orig kBET"])
        kbet_df["scaled orig kBET"] = 1 - kbet_df["orig kBET"]
        kbet_df["graph_conn"] = graph_conn
        kbet_df["PCR"] = 1 - pcr

        self.scores = scores
        self.kbet_df = kbet_df
        return self

    def save(self, path: str = None, tag: str = "count_k_90"):
        """
        Save per-cell-type and micro/macro aggregated scores.

        Parameters
        ----------
        path : str, optional
            Output directory. Defaults to ``self.sc_dir``.
        tag : str
            Suffix appended to output file names.
        """
        path = path or self.sc_dir
        os.makedirs(path, exist_ok=True)
        self.save_scores_per_celltype(
            self.scores, self.valid_cells, self.kbet_df, path, tag
        )
        self.save_scores_per_celltype_micro_macro(
            self.scores, self.valid_cells, self.kbet_df, path, f"micro_macro_{tag}"
        )
        return self

    # ------------------------------------------------------------------
    # Core metric methods
    # ------------------------------------------------------------------

    def distr_based_metrics(
        self,
        scores: pd.DataFrame,
        local_dist: pd.DataFrame,
        global_dist: pd.DataFrame,
        n_batches: int,
    ) -> pd.DataFrame:
        """
        Compute Jensen-Shannon divergence between local and global
        distributions for each cell.

        Parameters
        ----------
        scores : pd.DataFrame
            Results frame (cell IDs as index).
        local_dist : pd.DataFrame
            Local batch distribution per cell.
        global_dist : pd.DataFrame
            Global batch distribution per cell.
        n_batches : int
            Number of unique batches.

        Returns
        -------
        pd.DataFrame
            ``scores`` with an added ``"JS Dist"`` column.
        """
        scores["JS Dist"] = 0.0
        for cell_id in scores.index:
            loc = np.array(local_dist.loc[cell_id])
            glob = np.array(global_dist.loc[cell_id])
            scores.at[cell_id, "JS Dist"] = self._js(loc, glob)
        return scores

    def compute_intra_inter_distances(
        self, adata=None, batch_key: str = "Batch", label_key: str = "Group"
    ) -> pd.DataFrame:
        """
        Compute per-cell mean intra- and inter-batch distances in PCA space.

        Parameters
        ----------
        adata : AnnData, optional
            Defaults to ``self.adata``.
        batch_key : str
            Obs column for batch.
        label_key : str
            Obs column for cell type.

        Returns
        -------
        pd.DataFrame
            Columns: ``intra_mean``, ``inter_mean``, ``intra_inter_ratio``.
        """
        if adata is None:
            adata = self.adata

        X = adata.obsm["X_pca"]
        obs_df = adata.obs[[batch_key, label_key]].copy()
        obs_df["intra_mean"] = 0.0
        obs_df["inter_mean"] = 0.0

        for idx in range(len(adata)):
            cell_batch = obs_df.iloc[idx][batch_key]
            cell_group = obs_df.iloc[idx][label_key]

            same_group = obs_df[label_key] == cell_group
            same_batch = obs_df[batch_key] == cell_batch
            diff_batch = obs_df[batch_key] != cell_batch

            intra_indices = np.where(same_group & same_batch)[0]
            intra_indices = intra_indices[intra_indices != idx]

            if len(intra_indices) > 0:
                intra_mean_val = cdist([X[idx]], X[intra_indices], metric="euclidean")[0].mean()
                obs_df.iloc[idx, obs_df.columns.get_loc("intra_mean")] = intra_mean_val
            else:
                intra_mean_val = 0.0

            inter_indices = np.where(same_group & diff_batch)[0]
            if len(inter_indices) > 0:
                obs_df.iloc[idx, obs_df.columns.get_loc("inter_mean")] = (
                    cdist([X[idx]], X[inter_indices], metric="euclidean")[0].mean()
                )
            else:
                obs_df.iloc[idx, obs_df.columns.get_loc("inter_mean")] = intra_mean_val

        obs_df["intra_inter_ratio"] = np.where(
            (obs_df["intra_mean"] == 0) | (obs_df["inter_mean"] == 0),
            np.nan,
            obs_df["intra_mean"] / obs_df["inter_mean"],
        )
        return obs_df

    def conventional_metrics(
        self, int_df, batches_df, k, adata, batch_key, group_key, embed="X_pca"
    ):
        """
        Compute iLISI, kBET (scib), ASW batch, PCR, and graph connectivity.

        Parameters
        ----------
        int_df : pd.DataFrame
            Integrated embedding.
        batches_df : pd.DataFrame
            Batch labels.
        k : int
            Neighborhood size.
        adata : AnnData
        batch_key, group_key : str
        embed : str

        Returns
        -------
        tuple
            ``(ilisi_df, scib_kbet, sil_df, pcr, graph_conn)``
        """
        ilisi_df = self.compute_lisi_score(int_df, batches_df, k)

        scib_kbet = scib.me.kBET(
            adata, batch_key=batch_key, label_key=group_key,
            type_="full", embed=embed, return_df=True,
        )

        asw, sil_means, sil_df = silhouette_batch(
            adata, batch_key=batch_key, group_key=group_key,
            embed=embed, return_all=True,
        )

        pcr = scib.me.pcr(adata, covariate=batch_key, embed=embed)
        graph_conn = graph_connectivity_per_celltype(adata, label_key=group_key)

        return ilisi_df, scib_kbet, sil_df, pcr, graph_conn

    def compute_lisi_score(self, int_df, batches_df, k) -> pd.DataFrame:
        """
        Compute iLISI using the R ``lisi`` package.

        Parameters
        ----------
        int_df : pd.DataFrame
            Integrated embedding.
        batches_df : pd.DataFrame
            Batch labels.
        k : int
            Neighborhood size (perplexity = k / 3).

        Returns
        -------
        pd.DataFrame
            iLISI scores.
        """
        pandas2ri.activate()
        lisi = importr("lisi")

        r_int_df = pandas2ri.py2rpy(int_df)
        r_batches_df = ro.r["data.frame"](pandas2ri.py2rpy(batches_df))
        r_colnames = ro.StrVector(batches_df.columns)

        iLISI = lisi.compute_lisi(r_int_df, r_batches_df, r_colnames, perplexity=k / 3)
        iLISI_df = pandas2ri.rpy2py(iLISI)

        pandas2ri.deactivate()
        return iLISI_df

    def orig_kbet_score_k(
        self, adata, batch_key: str, group_key: str, sc_order: str, embed: str = "X_pca"
    ):
        """
        Compute original kBET score and neighborhood size for each cell type.

        Parameters
        ----------
        adata : AnnData
        batch_key, group_key : str
        sc_order : str
            Output directory identifier.
        embed : str

        Returns
        -------
        dict
            ``{cell_type: kbet_score}``
        dict
            ``{cell_type: neighborhood_size}``
        """
        orig_kbet = {}
        orig_kbet_k = {}

        for celltype in adata.obs[group_key].unique():
            mask = adata.obs[group_key] == celltype
            batches_celltype = np.array(adata.obs.loc[mask, batch_key])
            pca_coords_celltype = adata.obsm[embed][mask]

            score, k_kbet = self._orig_kbet_celltype(
                mtrx=pca_coords_celltype,
                batch=batches_celltype,
                celltype=celltype,
                sc_order=sc_order,
            )
            orig_kbet[celltype] = score
            orig_kbet_k[celltype] = k_kbet

        return orig_kbet, orig_kbet_k

    # ------------------------------------------------------------------
    # Distribution builders
    # ------------------------------------------------------------------

    def build_distribution(
        self, adata, dknn_df, celltypes_df, batches_df,
        dist_type: str = "count", weights_df=None,
    ) -> pd.DataFrame:
        """
        Build local or global batch distribution for each cell.

        Parameters
        ----------
        adata : AnnData
        dknn_df : pd.DataFrame
            kNN indices.
        celltypes_df, batches_df : pd.DataFrame
        dist_type : {"count", "weighted", "global"}
        weights_df : pd.DataFrame, optional
            Required when ``dist_type="weighted"``.

        Returns
        -------
        pd.DataFrame
        """
        k = len(dknn_df.columns)
        celltype_counts = adata.obs["Group"].value_counts()

        batches_neighbors_df = pd.DataFrame(index=dknn_df.index, columns=range(k))
        celltypes_neighbors_df = pd.DataFrame(index=dknn_df.index, columns=range(k))

        batches = adata[batches_neighbors_df.index].obs["Batch"].values
        celltypes = adata[celltypes_neighbors_df.index].obs["Group"].values

        for i in range(len(batches_neighbors_df.index)):
            batches_neighbors_df.iloc[i] = batches[dknn_df.iloc[i].values]
            celltypes_neighbors_df.iloc[i] = celltypes[dknn_df.iloc[i].values]

        dist = pd.DataFrame(0, index=batches_df.index, columns=batches.unique(), dtype=float)

        if dist_type == "global":
            for b in batches.unique():
                for cell_id in celltypes_neighbors_df.index:
                    cell_type = celltypes_df.iloc[cell_id]["Group"]
                    dist[b][cell_id] = adata[
                        (adata.obs.Batch == b) & (adata.obs.Group == cell_type)
                    ].shape[0]
            return dist

        for b in batches.unique():
            for cell_id in celltypes_neighbors_df.index:
                cell_type = celltypes_df.iloc[cell_id]["Group"]
                k_adjusted = min(k, celltype_counts[cell_type])

                neigh_celltypes = np.array(celltypes_neighbors_df.loc[cell_id].iloc[:k_adjusted])
                neigh_batch_labels = np.array(batches_neighbors_df.loc[cell_id].iloc[:k_adjusted])
                same_type_batch_mask = (neigh_celltypes == cell_type) & (neigh_batch_labels == b)

                if dist_type == "weighted":
                    neigh_weights = weights_df.loc[cell_id].values[:k_adjusted]
                    dist[b][cell_id] = neigh_weights[same_type_batch_mask].sum()
                elif dist_type == "count":
                    dist[b][cell_id] = same_type_batch_mask.sum()

        return dist


    def sBEE(
        self,
        df: pd.DataFrame,
        js_dist_key: str = "JS Dist",
        ratio_key: str = "intra_inter_ratio",
        sensitivity: float = 0.15,
    ) -> pd.DataFrame:
        """
        Compute the single-cell Batch Effect Evaluator (sBEE) score.

        Harmonic mean of the JS divergence score and the intra/inter-batch
        distance ratio score. Higher is better (1 = perfect integration).
        """
        js_score = 1 - df[js_dist_key]
        ratio_score = np.exp(-np.abs(1 - df[ratio_key]) / sensitivity)
        df["js_part"] = js_score
        df["ratio_part"] = ratio_score
        df["sBEE"] = 2 * js_score * ratio_score / (js_score + ratio_score)
        return df

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def save_scores_per_celltype(
        self, scores, valid_cells, kbet_df, path: str, tag: str = "count"
    ):
        """Save mean scores grouped by cell type to CSV."""
        scores_filtered = scores.loc[valid_cells]
        avg_df = scores_filtered.groupby("Cell type", observed=False).mean(numeric_only=True)
        avg_df = pd.concat([avg_df, kbet_df], axis=1)
        avg_df.to_csv(f"{path}/scores_per_celltype_{tag}.csv", index=True)

    def save_scores_per_celltype_micro_macro(
        self, scores, valid_cells, kbet_df, path: str, tag: str = "micro_macro"
    ):
        """Save micro and macro averaged scores grouped by cell type to CSV."""
        scores_filtered = scores.loc[valid_cells]
        numeric_cols = scores_filtered.select_dtypes(include="number").columns

        macro_df = (
            scores_filtered
            .groupby(["Cell type", "Batch"], observed=False)[numeric_cols].mean()
            .groupby("Cell type", observed=False).mean()
        )
        macro_df.columns = [f"{c}_macro" for c in macro_df.columns]

        micro_df = scores_filtered.groupby("Cell type", observed=False)[numeric_cols].mean()
        micro_df.columns = [f"{c}_micro" for c in micro_df.columns]

        avg_df = pd.concat([macro_df, micro_df, kbet_df], axis=1)
        interleaved = [col for c in numeric_cols for col in (f"{c}_macro", f"{c}_micro")]
        avg_df = avg_df[interleaved + list(kbet_df.columns)]
        avg_df.to_csv(f"{path}/scores_per_celltype_{tag}.csv", index=True)

    def save_scores_per_batch(
        self, scores, valid_cells, kbet_df, path: str, tag: str = "count"
    ):
        """Save mean scores grouped by batch to CSV."""
        scores_filtered = scores.loc[valid_cells]
        avg_df = scores_filtered.groupby("Batch", observed=False).mean(numeric_only=True)
        avg_df.to_csv(f"{path}/scores_per_batch_{tag}.csv", index=True)

    # ------------------------------------------------------------------
    # Static / pure helpers
    # ------------------------------------------------------------------

    @staticmethod
    def knn(
        df: pd.DataFrame,
        k: int = 90,
        metric: str = "euclidean",
        include_self: bool = True,
    ):
        """
        k-nearest neighbor search.

        Parameters
        ----------
        df : pd.DataFrame
            Rows = cells, columns = features.
        k : int
            Number of neighbors.
        metric : str
            Distance metric (e.g. ``"euclidean"``, ``"cosine"``).
        include_self : bool
            Whether to include the query cell as its own first neighbor.

        Returns
        -------
        indices : np.ndarray, shape (n_cells, k)
        distances : np.ndarray, shape (n_cells, k)
        """
        _k = k if include_self else k + 1
        X = df.to_numpy(dtype=np.float32, copy=False)
        nn = NearestNeighbors(n_neighbors=_k, metric=metric, algorithm="auto", n_jobs=-1).fit(X)
        distances, indices = nn.kneighbors(X, return_distance=True)
        if not include_self:
            distances, indices = distances[:, 1:], indices[:, 1:]
        return indices, distances

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _js(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """Jensen-Shannon distance between two distributions."""
        p = np.clip(p, epsilon, None)
        q = np.clip(q, epsilon, None)
        p = p / p.sum()
        q = q / q.sum()
        return jensenshannon(p, q, base=2.0)

    def _orig_kbet_celltype(self, mtrx, batch, celltype: str, sc_order: str):
        """
        Run the R kBET package for one cell type and save rejection status.

        Parameters
        ----------
        mtrx : array-like
            PCA coordinates for the cell type.
        batch : array-like
            Batch labels for the cell type.
        celltype : str
        sc_order : str
            Output directory.

        Returns
        -------
        score : float
        k : int
        """
        ro.r("library(kBET)")
        anndata2ri.activate()

        ro.globalenv["data_mtrx"] = mtrx
        ro.globalenv["batch"] = batch
        ro.r(
            "batch.estimate <- kBET("
            "  data_mtrx, batch, do.pca=FALSE, verbose=TRUE, plot=FALSE"
            ")"
        )

        score = ro.r("batch.estimate$summary$kBET.observed")[0]
        k = ro.r("batch.estimate$params$k0")[0]

        ro.r('source("orig_kbet_rejection_with_neighbors.R")')
        rejection_fn = ro.r["rejection_with_neighbors"]
        rejection_fn(
            df=mtrx,
            batch=batch,
            celltype=celltype,
            k0=int(k),
            sc_dir=f"{sc_order}/",
        )

        anndata2ri.deactivate()
        return score, k
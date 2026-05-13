import os
import pickle
import logging
import hashlib
import json
import tempfile
import time
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Set, Union

logger = logging.getLogger(__name__)

_DATASET_CACHE_VERSION = 1
_DATASET_CACHE_WAIT_SECONDS = 2.0
_DATASET_CACHE_STALE_LOCK_SECONDS = 60.0 * 30.0


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


class OpenVidDataset(Dataset):
    """
    Dataset for loading OpenVid-1M video-text pairs.
    
    Can work in two modes:
    1. Pre-processed mode: Load from pickle files (faster, requires pre-processing)
    2. Raw mode: Load videos and encode on-the-fly (slower, but no pre-processing needed)
    """
    
    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        preprocessed_dir: Optional[str] = None,
        use_preprocessed: bool = True,
        max_samples: Optional[int] = None,
        modality_filter: Optional[Union[str, List[str], Set[str]]] = None,
    ):
        """
        Initialize OpenVid-1M dataset.
        
        Args:
            csv_path: Path to OpenVid-1M.csv file
            video_dir: Directory containing video files
            preprocessed_dir: Directory containing pre-processed pickle files (if use_preprocessed=True)
            use_preprocessed: If True, load from pickle files. If False, load videos directly.
            max_samples: Maximum number of samples to load (for testing)
            modality_filter: Optional modality filter ("video"/"image" or list of values).
                Useful for building separate dataloaders for joint iv training.
        """
        self.video_dir = video_dir
        self.preprocessed_dir = preprocessed_dir
        self.use_preprocessed = use_preprocessed
        self.rank = int(os.environ.get("RANK", "0") or 0)
        self.dataset_cache_enabled = _env_flag("MOBILEOV_DATASET_CACHE", True)
        self.trust_preprocessed_manifest = _env_flag("MOBILEOV_TRUST_PREPROCESSED_MANIFEST", True)
        self.modality_filter: Optional[Set[str]] = None
        if modality_filter is not None:
            if isinstance(modality_filter, str):
                values = [modality_filter]
            else:
                values = list(modality_filter)
            normalized = {str(v).strip().lower() for v in values if str(v).strip()}
            self.modality_filter = normalized if normalized else None
        cache_path = self._get_cache_path(
            csv_path=csv_path,
            video_dir=video_dir,
            preprocessed_dir=preprocessed_dir,
            max_samples=max_samples,
        )
        self.data = self._load_or_build_index(
            csv_path=csv_path,
            video_dir=video_dir,
            preprocessed_dir=preprocessed_dir,
            max_samples=max_samples,
            cache_path=cache_path,
        )
        
        if self.modality_filter is None:
            logger.info(f"Loaded {len(self.data)} valid samples from OpenVid-style CSV")
        else:
            logger.info(
                "Loaded %d valid samples from OpenVid-style CSV (modality_filter=%s)",
                len(self.data),
                sorted(self.modality_filter),
            )

    def _get_cache_path(
        self,
        csv_path: str,
        video_dir: str,
        preprocessed_dir: Optional[str],
        max_samples: Optional[int],
    ) -> Optional[str]:
        if not self.dataset_cache_enabled:
            return None
        csv_abs = os.path.abspath(csv_path)
        try:
            csv_stat = os.stat(csv_abs)
            csv_sig = {"size": int(csv_stat.st_size), "mtime_ns": int(csv_stat.st_mtime_ns)}
        except OSError:
            csv_sig = {"size": -1, "mtime_ns": -1}
        cache_root = os.environ.get(
            "MOBILEOV_DATASET_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "mobileov_dataset_cache"),
        )
        cache_key = {
            "version": _DATASET_CACHE_VERSION,
            "csv_path": csv_abs,
            "csv_sig": csv_sig,
            "video_dir": os.path.abspath(video_dir),
            "preprocessed_dir": os.path.abspath(preprocessed_dir) if preprocessed_dir else "",
            "use_preprocessed": bool(self.use_preprocessed),
            "max_samples": int(max_samples) if max_samples is not None else None,
            "modality_filter": sorted(self.modality_filter) if self.modality_filter is not None else None,
            "trust_preprocessed_manifest": bool(self.trust_preprocessed_manifest),
        }
        key_text = json.dumps(cache_key, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(key_text.encode("utf-8")).hexdigest()[:16]
        os.makedirs(cache_root, exist_ok=True)
        return os.path.join(cache_root, f"openvid_index_{digest}.pkl")

    def _load_or_build_index(
        self,
        csv_path: str,
        video_dir: str,
        preprocessed_dir: Optional[str],
        max_samples: Optional[int],
        cache_path: Optional[str],
    ) -> List[Dict[str, object]]:
        lock_fd = None
        lock_path = f"{cache_path}.lock" if cache_path else None
        if cache_path:
            cached = self._try_load_cache(cache_path)
            if cached is not None:
                return cached
            lock_fd = self._try_acquire_lock(lock_path)
            if lock_fd is None:
                cached = self._wait_for_cache(cache_path, lock_path)
                if cached is not None:
                    return cached
                lock_fd = self._try_acquire_lock(lock_path)

        try:
            data = self._build_index_from_csv(
                csv_path=csv_path,
                video_dir=video_dir,
                preprocessed_dir=preprocessed_dir,
                max_samples=max_samples,
            )
            if cache_path and lock_fd is not None:
                self._write_cache(cache_path, data)
            return data
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass

    def _try_load_cache(self, cache_path: str) -> Optional[List[Dict[str, object]]]:
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if not isinstance(payload, dict) or payload.get("version") != _DATASET_CACHE_VERSION:
                logger.warning("Ignoring stale dataset cache at %s", cache_path)
                return None
            data = payload.get("data")
            if not isinstance(data, list):
                logger.warning("Ignoring malformed dataset cache at %s", cache_path)
                return None
            logger.info("Loaded dataset index cache from %s (%d samples)", cache_path, len(data))
            return data
        except Exception as exc:
            logger.warning("Failed to load dataset cache %s: %s", cache_path, exc)
            return None

    def _try_acquire_lock(self, lock_path: Optional[str]) -> Optional[int]:
        if not lock_path:
            return None
        try:
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None

    def _wait_for_cache(self, cache_path: str, lock_path: Optional[str]) -> Optional[List[Dict[str, object]]]:
        if not lock_path:
            return None
        wait_logged = False
        while True:
            cached = self._try_load_cache(cache_path)
            if cached is not None:
                return cached
            if not os.path.exists(lock_path):
                return None
            if not wait_logged:
                logger.info("Waiting for dataset cache builder to finish: %s", cache_path)
                wait_logged = True
            try:
                lock_age = time.time() - os.path.getmtime(lock_path)
                if lock_age > _DATASET_CACHE_STALE_LOCK_SECONDS:
                    logger.warning("Removing stale dataset cache lock: %s", lock_path)
                    os.unlink(lock_path)
                    return None
            except FileNotFoundError:
                return None
            time.sleep(_DATASET_CACHE_WAIT_SECONDS)

    def _write_cache(self, cache_path: str, data: List[Dict[str, object]]) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp_openvid_index_", suffix=".pkl", dir=os.path.dirname(cache_path))
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump({"version": _DATASET_CACHE_VERSION, "data": data}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            logger.info("Saved dataset index cache to %s (%d samples)", cache_path, len(data))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        logger.info("Loading OpenVid-1M CSV from %s", csv_path)
        try:
            try:
                return pd.read_csv(csv_path, quoting=1, escapechar="\\")
            except Exception:
                try:
                    return pd.read_csv(csv_path, quoting=1)
                except Exception:
                    return pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise

    def _build_index_from_csv(
        self,
        csv_path: str,
        video_dir: str,
        preprocessed_dir: Optional[str],
        max_samples: Optional[int],
    ) -> List[Dict[str, object]]:
        df = self._load_csv(csv_path)

        has_caption_col = "caption" in df.columns
        has_video_col = "video" in df.columns
        has_preprocessed_path_col = "preprocessed_path" in df.columns
        has_sample_idx_col = "sample_idx" in df.columns
        if not has_caption_col:
            raise ValueError("CSV file must contain 'caption' column")
        self.direct_preprocessed_mode = bool(
            self.use_preprocessed and (has_preprocessed_path_col or has_sample_idx_col)
        )
        if not has_video_col and not self.direct_preprocessed_mode:
            raise ValueError(
                "CSV missing required columns for legacy mode. Need either "
                "['video','caption'] or ['caption','preprocessed_path'] / ['caption','sample_idx']"
            )

        if self.direct_preprocessed_mode:
            return self._build_direct_preprocessed_index(df, video_dir, preprocessed_dir, max_samples)
        return self._build_legacy_index(df, video_dir, preprocessed_dir, max_samples)

    def _build_direct_preprocessed_index(
        self,
        df: pd.DataFrame,
        video_dir: str,
        preprocessed_dir: Optional[str],
        max_samples: Optional[int],
    ) -> List[Dict[str, object]]:
        work = df.copy()
        work["caption"] = work["caption"].fillna("").astype(str).str.strip()
        work = work[work["caption"] != ""].copy()

        if "preprocessed_path" in work.columns:
            work["preprocessed_path"] = work["preprocessed_path"].fillna("").astype(str).str.strip()
        else:
            work["preprocessed_path"] = ""

        if "sample_idx" in work.columns:
            work["sample_idx"] = pd.to_numeric(work["sample_idx"], errors="coerce")
        else:
            work["sample_idx"] = np.nan

        if preprocessed_dir:
            missing_pre = work["preprocessed_path"] == ""
            if missing_pre.any():
                work.loc[missing_pre, "preprocessed_path"] = work.loc[missing_pre, "sample_idx"].map(
                    lambda x: os.path.join(preprocessed_dir, f"sample_{int(x):08d}.pkl") if pd.notna(x) else ""
                )
            work["preprocessed_path"] = work["preprocessed_path"].map(
                lambda p: os.path.join(preprocessed_dir, p) if p and not os.path.isabs(p) else p
            )

        work = work[work["preprocessed_path"] != ""].copy()
        if not self.trust_preprocessed_manifest:
            work = work[work["preprocessed_path"].map(os.path.exists)].copy()

        if "video" in work.columns:
            work["video_name"] = work["video"].fillna("").astype(str).str.strip()
        else:
            work["video_name"] = ""
        for candidate_col in ("video_path", "media_path", "image_path", "source_id"):
            if candidate_col not in work.columns:
                continue
            candidate_series = work[candidate_col].fillna("").astype(str).str.strip()
            missing_mask = work["video_name"] == ""
            if missing_mask.any():
                work.loc[missing_mask, "video_name"] = candidate_series.loc[missing_mask].map(os.path.basename)
        missing_video_name = work["video_name"] == ""
        if missing_video_name.any():
            work.loc[missing_video_name, "video_name"] = work.loc[missing_video_name, "preprocessed_path"].map(
                os.path.basename
            )

        if "modality" in work.columns:
            work["modality"] = work["modality"].fillna("").astype(str).str.strip().str.lower()
        else:
            work["modality"] = ""
        missing_modality = work["modality"] == ""
        if missing_modality.any():
            work.loc[missing_modality, "modality"] = work.loc[missing_modality, "video_name"].map(
                lambda name: "image"
                if str(name).lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
                else "video"
            )
        if self.modality_filter is not None:
            work = work[work["modality"].isin(self.modality_filter)].copy()

        work["video_path"] = ""
        for candidate_col in ("video_path", "media_path", "image_path"):
            if candidate_col not in work.columns:
                continue
            candidate_series = work[candidate_col].fillna("").astype(str).str.strip()
            missing_path = work["video_path"] == ""
            if missing_path.any():
                work.loc[missing_path, "video_path"] = candidate_series.loc[missing_path]
        if video_dir:
            missing_path = work["video_path"] == ""
            if missing_path.any():
                work.loc[missing_path, "video_path"] = work.loc[missing_path, "video_name"].map(
                    lambda name: os.path.join(video_dir, name) if name else ""
                )

        missing_sample_idx = work["sample_idx"].isna()
        if missing_sample_idx.any():
            work.loc[missing_sample_idx, "sample_idx"] = np.arange(len(work))[missing_sample_idx.to_numpy()]
        work["sample_idx"] = work["sample_idx"].astype(int)

        keep_cols = ["video_path", "video_name", "caption", "preprocessed_path", "sample_idx", "modality"]
        work = work[keep_cols]
        if max_samples is not None:
            work = work.iloc[: int(max_samples)].copy()
        data = work.to_dict("records")
        logger.info(
            "Built direct-preprocessed dataset index with %d samples (trust_manifest=%s)",
            len(data),
            self.trust_preprocessed_manifest,
        )
        return data

    def _build_legacy_index(
        self,
        df: pd.DataFrame,
        video_dir: str,
        preprocessed_dir: Optional[str],
        max_samples: Optional[int],
    ) -> List[Dict[str, object]]:
        preprocessed_index = None
        if self.use_preprocessed and preprocessed_dir:
            try:
                files = [f for f in os.listdir(preprocessed_dir) if f.endswith(".pkl")]
                preprocessed_index = set()
                for f in files:
                    stem = os.path.splitext(f)[0]
                    preprocessed_index.add(stem)
                    if stem.endswith("_features"):
                        preprocessed_index.add(stem[: -len("_features")])
                logger.info(f"Found {len(preprocessed_index)} preprocessed samples in {preprocessed_dir}")
            except Exception as e:
                logger.warning(f"Failed to scan preprocessed dir {preprocessed_dir}: {e}")
                preprocessed_index = None

        data: List[Dict[str, object]] = []
        for idx, row in enumerate(df.itertuples(index=False), start=0):
            if idx % 100000 == 0 and idx > 0:
                logger.info(f"Scanned {idx} rows, kept {len(data)} samples so far")
            if max_samples and len(data) >= max_samples:
                break

            caption_val = getattr(row, "caption", None)
            if pd.isna(caption_val):
                logger.warning(f"Skipping row {idx}: missing caption")
                continue
            caption = str(caption_val).strip()

            video_val = getattr(row, "video", None)
            if pd.isna(video_val):
                logger.warning(f"Skipping row {idx}: missing video")
                continue
            video_name = str(video_val).strip()
            row_modality = str(getattr(row, "modality", "") or "").strip().lower()
            if not row_modality:
                row_modality = "image" if video_name.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                ) else "video"
            if self.modality_filter is not None and row_modality not in self.modality_filter:
                continue

            if video_name.lower() == "video" or video_name == "video,caption,aesthetic score,motion score,temporal consistency score,camera motion,frame,fps,seconds":
                logger.warning(f"Skipping row {idx}: appears to be header row")
                continue
            if len(video_name) > 255:
                logger.warning(f"Skipping row {idx}: video name too long ({len(video_name)} chars): {video_name[:50]}...")
                continue
            if "," in video_name and len(video_name) > 100:
                logger.warning(f"Skipping row {idx}: video name contains comma and is too long, likely malformed CSV row")
                continue

            if self.use_preprocessed and preprocessed_dir:
                video_basename = os.path.splitext(video_name)[0]
                candidate_names = [
                    f"{video_basename}_features.pkl",
                    f"{video_name}_features.pkl",
                    f"{video_basename}.pkl",
                    f"{video_name}.pkl",
                ]
                if preprocessed_index is not None:
                    if video_basename not in preprocessed_index and video_name not in preprocessed_index:
                        continue
                preprocessed_path = None
                for candidate_name in candidate_names:
                    candidate_path = os.path.join(preprocessed_dir, candidate_name)
                    if os.path.exists(candidate_path):
                        preprocessed_path = candidate_path
                        break
                if preprocessed_path is None:
                    continue

                video_path = os.path.join(video_dir, video_name)
                if not os.path.exists(video_path):
                    video_path = os.path.join(video_dir, f"{video_name}.mp4")
                    if not os.path.exists(video_path):
                        video_path = None

                data.append(
                    {
                        "video_path": video_path,
                        "video_name": video_name,
                        "caption": caption,
                        "preprocessed_path": preprocessed_path,
                        "sample_idx": int(idx),
                        "modality": row_modality,
                    }
                )
            else:
                video_path = os.path.join(video_dir, video_name)
                if not os.path.exists(video_path):
                    video_path = os.path.join(video_dir, f"{video_name}.mp4")
                    if not os.path.exists(video_path):
                        continue

                data.append(
                    {
                        "video_path": video_path,
                        "video_name": video_name,
                        "caption": caption,
                        "preprocessed_path": None,
                        "sample_idx": int(idx),
                        "modality": row_modality,
                    }
                )
        return data
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            dict: Dictionary containing all required inputs for the model
        """
        if len(self.data) == 0:
            logger.error("Dataset is empty!")
            return {
                'prompt': 'Dataset is empty',
                'latent_feature': torch.zeros(1, 16, 21, 32, 32),
            }
        
        real_idx = idx % len(self.data)  # Ensure valid index
        try_idx = 0
        max_retries = min(20, len(self.data))  # Don't retry more than dataset size
        
        while try_idx < max_retries:
            try:
                item = self.data[real_idx]
                
                if self.use_preprocessed and item.get('preprocessed_path') and os.path.exists(item['preprocessed_path']):
                    # Load from preprocessed pickle file
                    with open(item['preprocessed_path'], 'rb') as f:
                        data = pickle.load(f)
                    
                    # Ensure required keys
                    if 'prompt' not in data:
                        data['prompt'] = item.get('caption', '')
                    
                    # Convert to tensors if needed
                    for key in data:
                        if hasattr(data[key], 'shape') and not isinstance(data[key], torch.Tensor):
                            data[key] = torch.tensor(data[key])
                    if "sample_idx" not in data:
                        data["sample_idx"] = int(item.get("sample_idx", real_idx))
                    else:
                        try:
                            data["sample_idx"] = int(data["sample_idx"])
                        except Exception:
                            data["sample_idx"] = int(item.get("sample_idx", real_idx))
                    if "modality" not in data:
                        data["modality"] = item.get("modality", "video")
                    
                    return data
                else:
                    # Raw mode: return video path and caption (will need on-the-fly encoding)
                    # Note: This mode requires additional processing in training loop
                    return {
                        'video_path': item.get('video_path'),
                        'prompt': item.get('caption', ''),
                        'video_name': item.get('video_name', ''),
                        'sample_idx': int(real_idx),
                        'modality': item.get('modality', 'video'),
                    }
                    
            except Exception as e:
                video_name = item.get('video_name', 'unknown') if 'item' in locals() else 'unknown'
                logger.warning(f"Error loading sample {real_idx} (video: {video_name[:50]}...): {str(e)[:200]}")
                try_idx += 1
                if try_idx < max_retries:
                    # Try next index instead of random to avoid infinite loops
                    real_idx = (real_idx + 1) % len(self.data)
                    logger.info(f"Retrying with next sample (try {try_idx + 1}/{max_retries})")
        
        # Fallback: return dummy data (should never reach here if dataset is valid)
        logger.error(f"Failed to load any sample after {try_idx} tries. Dataset size: {len(self.data)}")
        return {
            'prompt': 'Error loading sample',
            'latent_feature': torch.zeros(1, 16, 21, 32, 32),  # Dummy shape
            'sample_idx': int(real_idx),
        }


def openvid_collate_fn(batch):
    """
    Custom collate function for OpenVidDataset.
    Similar to omnivideo_collate_fn but handles OpenVid-1M format.
    """
    # Filter out None values
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise ValueError("Batch is empty after filtering None values")
    
    elem = batch[0]
    if elem is None:
        raise ValueError("First element in batch is None")
    
    result = {}
    
    for key in elem:
        if key == 'prompt':
            result[key] = [d[key] for d in batch]
        elif key == 'video_path' or key == 'video_name':
            # Keep as list
            result[key] = [d[key] for d in batch]
        elif key == "sample_idx":
            result[key] = torch.tensor([int(d[key]) for d in batch], dtype=torch.long)
        elif key == 'text_emb' or key == 'vlm_last_hidden_states':
            # Keep variable-length text embeddings as a list to avoid stacking errors.
            result[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            tensors = [d[key] for d in batch]
            shapes = [t.shape for t in tensors]
            if len(set(shapes)) == 1:
                result[key] = torch.stack(tensors, dim=0)
            else:
                try:
                    result[key] = torch.stack(tensors, dim=0)
                except RuntimeError as e:
                    if "stack expects each tensor to be equal size" in str(e):
                        # Use first tensor's shape as target and match each dimension.
                        target_shape = shapes[0]
                        corrected_tensors = []
                        for tensor in tensors:
                            if tensor.shape == target_shape:
                                corrected_tensors.append(tensor)
                                continue
                            fixed = tensor
                            for dim, target_dim in enumerate(target_shape):
                                cur_dim = fixed.shape[dim]
                                if cur_dim > target_dim:
                                    fixed = fixed.narrow(dim, 0, target_dim)
                                elif cur_dim < target_dim:
                                    if cur_dim <= 0:
                                        raise RuntimeError(
                                            f"Cannot pad empty tensor for key={key}, shape={tuple(tensor.shape)}"
                                        )
                                    pad_num = target_dim - cur_dim
                                    tail = fixed.select(dim, cur_dim - 1).unsqueeze(dim)
                                    repeat_shape = [1] * fixed.dim()
                                    repeat_shape[dim] = pad_num
                                    tail = tail.repeat(*repeat_shape)
                                    fixed = torch.cat([fixed, tail], dim=dim)
                            corrected_tensors.append(fixed)
                        result[key] = torch.stack(corrected_tensors, dim=0)
                    else:
                        raise e
        else:
            result[key] = [d[key] for d in batch]
    
    return result


def create_openvid_dataloader(
    csv_path: str,
    video_dir: str,
    preprocessed_dir: Optional[str] = None,
    use_preprocessed: bool = True,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    max_samples: Optional[int] = None,
):
    """
    Create DataLoader for OpenVid-1M dataset.
    
    Args:
        csv_path: Path to OpenVid-1M.csv
        video_dir: Directory containing video files
        preprocessed_dir: Directory with preprocessed pickle files
        use_preprocessed: Whether to use preprocessed features
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        distributed: Whether using distributed training
        rank: Process rank
        world_size: Total number of processes
        max_samples: Maximum samples to load (for testing)
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = OpenVidDataset(
        csv_path=csv_path,
        video_dir=video_dir,
        preprocessed_dir=preprocessed_dir,
        use_preprocessed=use_preprocessed,
        max_samples=max_samples,
    )
    
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=openvid_collate_fn
    )
    
    return dataloader

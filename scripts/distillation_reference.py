"""QBB-New sample200 generation/filter/validation, copied with pinned provenance.
Reference SHA256 is recorded in the distillation config; no QBB-New runtime dependency.
"""
from __future__ import annotations
import hashlib,struct,math,time
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as functional
from fluxbin_style import sha256_file, tensor_sha256
from fluxbin_style.evaluation import sha256_token_sequences,summed_cross_entropy_fp32

def sha256_text_sequence(values):
    digest=hashlib.sha256()
    for value in values:
        data=value.encode('utf-8');digest.update(struct.pack('>Q',len(data)));digest.update(data)
    return digest.hexdigest()

def finite_perplexity(mean_nll: float | None) -> float | None:
    if mean_nll is None:
        return None
    try:
        value = math.exp(mean_nll)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None

def generate_sequences(
    teacher: torch.nn.Module,
    *,
    sample_count: int,
    sequence_length: int,
    vocabulary_size: int,
    temperature: float,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    label: str,
) -> list[list[int]]:
    if sample_count <= 0 or sequence_length < 2 or batch_size <= 0:
        raise ValueError("invalid synthetic generation settings")
    sequences: list[list[int]] = []
    with torch.inference_mode():
        for start in range(0, sample_count, batch_size):
            count = min(batch_size, sample_count - start)
            tokens = torch.randint(
                vocabulary_size,
                (count, 1),
                generator=generator,
                device=device,
            )
            for _step in range(1, sequence_length):
                logits = teacher(
                    input_ids=tokens,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1, :]
                probabilities = functional.softmax(
                    logits.float() / temperature,
                    dim=-1,
                )
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
                tokens = torch.cat((tokens, next_token), dim=1)
                del logits, probabilities, next_token
            sequences.extend(tokens.cpu().tolist())
            print(
                f"QBB_NEW_DISTILL_{label}_GENERATION="
                f"{len(sequences)}/{sample_count}",
                flush=True,
            )
            del tokens
    return [[int(token) for token in sequence] for sequence in sequences]

def transformer_blocks(model: torch.nn.Module) -> torch.nn.ModuleList:
    blocks = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(blocks, torch.nn.ModuleList) or not blocks:
        raise ValueError("Qwen3 transformer blocks are unavailable")
    return blocks

def load_validation_protocol(
    config: dict[str, Any],
    dataset_path: Path,
    snapshot_root: Path,
) -> tuple[list[list[int]], dict[str, Any]]:
    from datasets import Dataset
    from transformers import AutoTokenizer

    settings = config["real_validation"]
    dataset_contract = settings["dataset"]
    if dataset_contract["split"] != "validation":
        raise ValueError("real validation must use the validation split")
    if not str(dataset_path).endswith(dataset_contract["relative_path"]):
        raise ValueError("validation dataset path drifted")
    dataset = Dataset.from_parquet(str(dataset_path))
    raw_lines = list(dataset[dataset_contract["field"]])
    protocol = settings["protocol"]
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_root,
        local_files_only=True,
        use_fast=bool(protocol["tokenizer_use_fast"]),
    )
    tokens = tokenizer.encode(
        protocol["join_separator"].join(raw_lines),
        add_special_tokens=bool(protocol["add_special_tokens"]),
    )
    sequence_length = int(protocol["sequence_length"])
    used_count = len(tokens) // sequence_length * sequence_length
    blocks = [
        tokens[start : start + sequence_length]
        for start in range(0, used_count, sequence_length)
    ]
    if not blocks:
        raise RuntimeError("validation protocol produced no full blocks")
    observed = {
        "dataset_sha256": sha256_file(dataset_path),
        "row_count": len(raw_lines),
        "raw_rows_sha256": sha256_text_sequence(raw_lines),
        "tokenizer_class": type(tokenizer).__name__,
        "token_count": len(tokens),
        "tokens_sha256": sha256_token_sequences([tokens]),
        "full_block_count": len(blocks),
        "blocks_sha256": sha256_token_sequences(blocks),
        "used_token_count": used_count,
        "dropped_tail_token_count": len(tokens) - used_count,
        "scored_transition_count": len(blocks)
        * int(protocol["scored_transitions_per_block"]),
    }
    return blocks, {**dataset_contract, **protocol, **observed}

def validation_metrics(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    blocks: list[list[int]],
    *,
    device: torch.device,
    logit_chunk_tokens: int = 128,
) -> dict[str, Any]:
    total_student_nll: list[float] = []
    total_teacher_nll: list[float] = []
    feature_losses: list[float] = []
    nonfinite_blocks: list[int] = []
    started = time.monotonic()
    student.eval()
    teacher.eval()
    teacher_blocks = transformer_blocks(teacher)
    student_blocks = transformer_blocks(student)
    with torch.inference_mode():
        for block_index, block in enumerate(blocks):
            teacher_features: list[torch.Tensor] = []
            student_features: list[torch.Tensor] = []

            def capture(target: list[torch.Tensor]):
                def hook(
                    _module: torch.nn.Module,
                    _inputs: tuple[torch.Tensor, ...],
                    output: torch.Tensor | tuple[torch.Tensor, ...],
                ) -> None:
                    target.append(output[0] if isinstance(output, tuple) else output)

                return hook

            handles = [
                layer.register_forward_hook(capture(teacher_features))
                for layer in teacher_blocks
            ] + [
                layer.register_forward_hook(capture(student_features))
                for layer in student_blocks
            ]
            input_ids = torch.tensor(block, dtype=torch.long, device=device).unsqueeze(0)
            try:
                teacher_logits = teacher(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=0,
                ).logits[0, :-1, :]
                student_logits = student(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=0,
                ).logits[0, :-1, :]
            finally:
                for handle in handles:
                    handle.remove()
            if len(teacher_features) != len(student_features):
                raise ValueError("validation feature capture drifted")
            feature_value = float(
                torch.stack(
                    [
                        functional.mse_loss(student_feature.float(), teacher_feature.float())
                        for teacher_feature, student_feature in zip(
                            teacher_features,
                            student_features,
                            strict=True,
                        )
                    ]
                ).mean()
            )
            labels = input_ids[0, 1:]
            student_chunks: list[float] = []
            teacher_chunks: list[float] = []
            for start in range(0, labels.numel(), logit_chunk_tokens):
                stop = min(start + logit_chunk_tokens, labels.numel())
                student_chunks.append(
                    float(
                        summed_cross_entropy_fp32(
                            student_logits[start:stop], labels[start:stop]
                        )
                    )
                )
                teacher_chunks.append(
                    float(
                        summed_cross_entropy_fp32(
                            teacher_logits[start:stop], labels[start:stop]
                        )
                    )
                )
            values = [*student_chunks, *teacher_chunks, feature_value]
            if all(math.isfinite(value) for value in values):
                total_student_nll.append(math.fsum(student_chunks))
                total_teacher_nll.append(math.fsum(teacher_chunks))
                feature_losses.append(feature_value)
            else:
                nonfinite_blocks.append(block_index)
            del input_ids, teacher_logits, student_logits, labels
            if (block_index + 1) % 10 == 0 or block_index + 1 == len(blocks):
                print(
                    f"QBB_NEW_DISTILL_REAL_VALIDATION="
                    f"{block_index + 1}/{len(blocks)}",
                    flush=True,
                )
    transition_count = sum(len(block) - 1 for block in blocks)
    student_total = math.fsum(total_student_nll) if not nonfinite_blocks else None
    teacher_total = math.fsum(total_teacher_nll) if not nonfinite_blocks else None
    student_mean = student_total / transition_count if student_total is not None else None
    teacher_mean = teacher_total / transition_count if teacher_total is not None else None
    student_ppl = finite_perplexity(student_mean)
    teacher_ppl = finite_perplexity(teacher_mean)
    return {
        "block_count": len(blocks),
        "scored_transition_count": transition_count,
        "student_total_nll": student_total,
        "student_mean_nll": student_mean,
        "student_perplexity": student_ppl,
        "teacher_total_nll": teacher_total,
        "teacher_mean_nll": teacher_mean,
        "teacher_perplexity": teacher_ppl,
        "feature_mse": (
            math.fsum(feature_losses) / len(feature_losses)
            if feature_losses
            else None
        ),
        "metrics_valid": (
            not nonfinite_blocks
            and student_ppl is not None
            and teacher_ppl is not None
        ),
        "nonfinite_block_indices": nonfinite_blocks,
        "elapsed_seconds": time.monotonic() - started,
    }

def filter_candidates(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    sequences: list[list[int]],
    *,
    keep_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], list[int]]:
    if not 0 < keep_count <= len(sequences):
        raise ValueError("invalid retained sample count")
    scores: list[float] = []
    teacher.eval()
    student.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            teacher_logits = teacher(
                input_ids=input_ids,
                use_cache=False,
                logits_to_keep=0,
            ).logits.float()
            student_logits = student(
                input_ids=input_ids,
                use_cache=False,
                logits_to_keep=0,
            ).logits.float()
            batch_scores = (student_logits - teacher_logits).square().mean(dim=(1, 2))
            scores.extend(float(value) for value in batch_scores)
            del input_ids, teacher_logits, student_logits, batch_scores
            print(
                f"QBB_NEW_DISTILL_FILTER={min(start + batch_size, len(sequences))}/"
                f"{len(sequences)}",
                flush=True,
            )
    selected_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:keep_count]
    return scores, selected_indices

"""Text-only prefix selection and conservative extra-pass runtime accounting."""
import math
import numpy as np

if __package__:
    from .span026_context import body_and_spans
else:
    from context_coverage import body_and_spans


def prefix_mask(records,tokenizer,batch_size=128):
    if batch_size<=0:raise ValueError('Positive tokenizer batch size required')
    selected=[]
    for start in range(0,len(records),batch_size):
        texts=[body_and_spans(row)[0] for row in records[start:start+batch_size]]
        encoded=tokenizer(texts,add_special_tokens=False,max_length=4097,truncation=True,padding=False)
        selected.extend(len(ids)>4096 for ids in encoded['input_ids'])
    return np.asarray(selected,dtype=bool)


def projection(parent_seconds,parent_g_uniform_seconds,worst_g_uniform_seconds,
               current_g_uniform_seconds,extended_seconds,extended_rows,changed_fraction,
               extended_warmup_seconds,full_selector_seconds,
               extra_parent_startup_seconds=0,target_rows=25000,base_rows=512):
    values=[parent_seconds,parent_g_uniform_seconds,worst_g_uniform_seconds,current_g_uniform_seconds,
            extended_seconds,extended_rows,changed_fraction,extended_warmup_seconds,
            full_selector_seconds,extra_parent_startup_seconds,target_rows,base_rows]
    if not all(math.isfinite(float(x)) and float(x)>=0 for x in values):raise ValueError('Invalid runtime value')
    if min(parent_seconds,parent_g_uniform_seconds,extended_rows,target_rows,base_rows)<=0:
        raise ValueError('Positive reference sizes/timing required')
    if not 0<changed_fraction<=1:raise ValueError('Invalid changed prevalence')
    margin=1.2
    slowdown=max(0,max(worst_g_uniform_seconds,current_g_uniform_seconds)-parent_g_uniform_seconds)
    baseline=parent_seconds+slowdown*target_rows/base_rows*margin+extra_parent_startup_seconds
    extra=extended_seconds/extended_rows*changed_fraction*target_rows*margin
    # Keep the measured FULL57477-row selector cost as overhead for a25k rerun;
    # do not hide this cost or scale it down with an unverified transfer factor.
    total=baseline+extra+extended_warmup_seconds+full_selector_seconds
    return dict(projection_seconds=total,retained_parent_projection_seconds=parent_seconds,
        conservative_parent_seconds=baseline,extra_full_pass_projection_seconds=extra,
        extended_warmup_seconds=extended_warmup_seconds,full_selector_seconds=full_selector_seconds,
        margin=margin,parent_includes_original_overhead_and_600s_reserve=True,
        divides_already_parallel_wall_time_by_gpu_count=False,
        engineering_screen_passed=total<32400,official_submission_eligible=False)

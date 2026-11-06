"""Clock/quota boundary for the separately authorized hidden evaluation allowance."""
from datetime import datetime


def admit_frozen_submission(plan,config,observed,now):
    q=observed['quota']
    if observed['nonterminal_jobs'] or q['reserved_seconds']!=0:raise ValueError('Another job/reservation remains active')
    if q['paid_scaling']:raise ValueError('Paid scaling is outside this contract')
    if q['refresh_time']!=plan['initial_quota_refresh_time']:raise ValueError('Quota refresh cannot authorize another admission')
    research=q['used_seconds']-plan['quota_used_at_start']
    if research<-.1 or research>config['gpu_session_seconds_cap']+.1:raise ValueError('Original research ceiling exceeded')
    if now+config['hidden_submission_seconds_cap']+300>=datetime.fromisoformat(config['deadline']).timestamp():raise ValueError('Full hidden allowance no longer fits original clock')
    return research

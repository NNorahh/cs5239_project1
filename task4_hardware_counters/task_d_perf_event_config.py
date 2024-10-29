# IA32_PERFEVENTSEL0 for enable L2_RQSTS.MISS performance counter

def get_perf_event_sel0():
    EN = 1 << 22         # Enable the counter (set bit 22 to 1)
    USR = 1 << 16        # Monitor only ring 3 (user mode) events (set bit 16 to 1)
    # According to Intel's developer manual, the UMASK=3FH, EVSEL=24H for L2_RQSTS.MISS event
    UMASK = 0x3F << 8    # UMASK for L2_RQSTS.MISS event (bits 8-15)
    EVSEL = 0x24         # Event Select for L2_RQSTS.MISS event (bits 0-7)

    final_value = EN | USR | UMASK | EVSEL
    print(f"Final value for IA32_PERFEVENTSEL0: 0x{final_value:08x}")
    return final_value

if __name__ == "__main__":
    get_perf_event_sel0()

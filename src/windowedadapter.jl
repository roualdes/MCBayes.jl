mutable struct OnlineWindowedAdapter
    closewindow::Int
    windowsize::Int
    const firstwindow::Int
    const lastwindow::Int
end

function OnlineWindowedAdapter(warmup; initbuffer = 75, termbuffer = 50, windowsize = 25)
    # TODO probably want some reasonable checks on warmup values
    return OnlineWindowedAdapter(
        initbuffer + windowsize,
        windowsize,
        initbuffer,
        warmup - termbuffer
    )
end

function calculate_nextwindow!(owa::OnlineWindowedAdapter)
    owa.windowsize *= 2
    nextclosewindow = owa.closewindow + owa.windowsize
    owa.closewindow = min(nextclosewindow, owa.lastwindow)
end

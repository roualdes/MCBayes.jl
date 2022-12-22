mutable struct WindowedAdaptationSchedule
    closewindow::Int
    windowsize::Int
    const warmup::Int
    const firstwindow::Int
    const lastwindow::Int
end

function WindowedAdaptationSchedule(warmup; initbuffer=75, termbuffer=50, windowsize=25)
    # TODO probably want some reasonable checks on warmup values
    return WindowedAdaptationSchedule(
        initbuffer + windowsize, windowsize, warmup, initbuffer, warmup - termbuffer
    )
end

function calculate_nextwindow!(ws::WindowedAdaptationSchedule)
    ws.windowsize *= 2
    nextclosewindow = ws.closewindow + ws.windowsize
    return ws.closewindow = if ws.closewindow + 2 * ws.windowsize > ws.lastwindow
        ws.lastwindow
    else
        min(nextclosewindow, ws.lastwindow)
    end
end

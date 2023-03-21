using MCBayes
using Documenter

DocMeta.setdocmeta!(MCBayes, :DocTestSetup, :(using MCBayes); recursive=true)

makedocs(;
    modules=[MCBayes],
    authors="Edward A. Roualdes",
    repo="https://github.com/roualdes/MCBayes.jl/blob/{commit}{path}#{line}",
    sitename="MCBayes.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://roualdes.github.io/MCBayes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md", "Design" => "design.md"],
)

deploydocs(; repo="github.com/roualdes/MCBayes.jl", devbranch="main")

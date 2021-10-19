module ElasticFusion

import Images


include("impl.jl")


export Fusion, process_frame!, render, save_ply

using ElasticFusionLib_jll


end # module

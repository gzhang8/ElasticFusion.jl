




mutable struct Fusion
    c_ptr::Ptr{Cvoid}
    confidence::Float32
end

function delete!(ef::Fusion)
    ccall((:delete_elastic_fusion, :libefusion), Cvoid, (Ptr{Cvoid},), ef.c_ptr)
end

function Fusion(;
    w::Union{Int64, Int32},
    h::Union{Int64, Int32},
    fx::Union{Float32, Float64},
    fy::Union{Float32, Float64},
    cx::Union{Float32, Float64},
    cy::Union{Float32, Float64},
    timeDelta::Union{Int64, Int32},
    countThresh::Union{Int64, Int32},
    errThresh::Union{Float32, Float64},
    covThresh::Union{Float32, Float64},
    closeLoops::Bool, #// bool
    iclnuim::Bool,  #// bool
    reloc::Bool, #// bool
    photoThresh::Union{Int64, Int32, Float32, Float64},
    confidence::Union{Int64, Int32, Float32, Float64},
    depthCut::Union{Int64, Int32, Float32, Float64},
    icpThresh::Union{Int64, Int32, Float32, Float64},
    fastOdom::Bool,       #// bool
    fernThresh::Union{Int64, Int32, Float32, Float64},
    so3::Bool, #// bool
    frameToFrameRGB::Bool, # // bool
    fileName::String
    )
    ef_ptr = ccall(
        (:create_elastic_fustion, :libefusion),
        Ptr{Cvoid},
        (Cint, Cint, Cfloat, Cfloat, Cfloat, Cfloat, # end cy
         Cint, Cint, Cfloat, Cfloat, Cint, Cint, Cint, # end reloc
         Cfloat, Cfloat, Cfloat, Cfloat, Cint, Cfloat, Cint, Cint,
         Cstring),
         Cint(w),
         Cint(h),
         Cfloat(fx),
         Cfloat(fy),
         Cfloat(cx),
         Cfloat(cy),
         Cint(timeDelta),
         Cint(countThresh),
         Cfloat(errThresh),
         Cfloat(covThresh),
         Cint(closeLoops),
         Cint(iclnuim),
         Cint(reloc),
         Cfloat(photoThresh),
         Cfloat(confidence),
         Cfloat(depthCut),
         Cfloat(icpThresh),
         Cint(fastOdom),
         Cfloat(fernThresh),
         Cint(so3),
         Cint(frameToFrameRGB),
         fileName
    )

    e = Fusion(ef_ptr, Float32(confidence))
    finalizer(delete!, e)
    return e
end






# void render(
#     void* efusion_void_ptr,
#     double* mvp_ptr,
#     float threshold,
#     int drawUnstable,//bool
#     int drawNormals, //bool
#     int drawColors,  //bool
#     int drawPoints,  //bool
#     int drawWindow,  //bool
#     int drawTimes,   //bool
#     int time,
#     int timeDelta)


function process_frame!(
    ef::Fusion,
    image::Matrix{Images.RGB{Images.Normed{UInt8,8}}},
    depth::Matrix{UInt16},
    timestamp::Int64;
    depth_factor::Union{Float32, Float64}=1000.0,
    pose::Matrix{Float32}=Matrix{Float32}(undef, 0, 0),
    weightMultiplier::Union{Float32, Float64}=1.0,
    bootstrap::Bool=false
    )
#     void process_frame(
#                void* efusion_void_ptr,
#                unsigned char * rgb,
#                unsigned short * depth,
#                int64_t timestamp,
#                float* in_pose_col_major,
#                float weightMultiplier,
#                int bootstrap // bool
#                )

    pose_ptr = if length(pose) == 0
        convert(Ptr{Cfloat}, C_NULL)
    else
        Base.unsafe_convert(Ptr{Cfloat}, pose)
    end
    h, w = size(depth)
    depth = permutedims(depth,  [2, 1])

    # image16 = reshape(reinterpret(UInt8, image), 3, h, w)

    # image16_row_major = permutedims(reshape(image16, (3, w, h)), [1, 3, 2])
    image16_row_major = permutedims(image, [2, 1])
    ccall(
        (:process_frame, :libefusion),
        Cvoid,
        (Ptr{Cvoid}, Ptr{Cuchar}, Ptr{Cushort}, Clonglong,
         Cfloat,
         Ptr{Cfloat}, Cfloat, Cint),
        ef.c_ptr,
        Base.unsafe_convert(Ptr{Cuchar}, image16_row_major),
        Base.unsafe_convert(Ptr{Cushort}, depth),
        timestamp,
        Cfloat(depth_factor),
        pose_ptr,
        Cfloat(weightMultiplier),
        Cint(bootstrap)
        )
end

function render(
    ef::Fusion,
    mvp::Matrix{Float64};
    threshold::Union{Int32, Int64, Float64, Float32},
    drawUnstable::Bool,#//bool
    drawNormals::Bool,#, //bool
    drawColors::Bool,#,  //bool
    drawPoints::Bool,#,  //bool
    drawWindow::Bool,#,  //bool
    drawTimes::Bool,#,   //bool
    time::Union{Int64, Int32},
    timeDelta::Union{Int64, Int32}
)
    ccall(
        (:render, :libefusion),
        Cvoid,
        (Ptr{Cvoid}, Ptr{Cdouble}, Cfloat,
         Cint, Cint, Cint, Cint, Cint, Cint, Cint, Cint),
        ef.c_ptr,
        Base.unsafe_convert(Ptr{Cdouble}, mvp),
        Cfloat(threshold),
        Cint(drawUnstable),
        Cint(drawNormals),#, //bool
        Cint(drawColors),#,  //bool
        Cint(drawPoints),#,  //bool
        Cint(drawWindow),#,  //bool
        Cint(drawTimes),#,   //bool
        Cint(time),
        Cint(timeDelta)
    )
end

"""
this one use the default confidence threshold to filter out temp ones
"""
function save_ply(
    ef::Fusion, ply_fpath::String; confidence_threshold::Real=ef.confidence
)


    ccall(
        (:save_ply, :libefusion),
        Cvoid,
        (Ptr{Cvoid}, Cstring, Cfloat),
        ef.c_ptr,
        ply_fpath,
        Cfloat(confidence_threshold)
    )

    # void* save_ply(
    #         void* efusion_void_ptr,
    #         char* ply_fpath){
end

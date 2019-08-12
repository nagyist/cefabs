//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2011 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
// 
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
#pragma once

#include "gpu_image.h"

#ifdef __CUDACC__

template<typename T> 
struct gpu_sampler_SRC1 {
    __host__ gpu_sampler_SRC1 (const gpu_image<T>& img, cudaTextureFilterMode filter_mode=cudaFilterModePoint) {
        s_texSRC1.filterMode = filter_mode;
        GPU_SAFE_CALL(cudaBindTexture2D(0, s_texSRC1, img.ptr(), img.w(), img.h(), img.pitch()));
    }

    __host__ ~gpu_sampler_SRC1 () {
        s_texSRC1.filterMode = cudaFilterModePoint;
        cudaUnbindTexture(s_texSRC1);
    }
    
    __device__ T operator()(float x, float y) const { 
        return tex2D(s_texSRC1, x, y); 
    } 
};


template<typename T> 
struct gpu_resampler_SRC1 {
    float2 s_;

    __host__ gpu_resampler_SRC1 (const gpu_image<T>& img, float2 s, cudaTextureFilterMode filter_mode=cudaFilterModePoint) {
        s_ = s;
        s_texSRC1.filterMode = filter_mode;
        GPU_SAFE_CALL(cudaBindTexture2D(0, s_texSRC1, img.ptr(), img.w(), img.h(), img.pitch()));
    }

    __host__ ~gpu_resampler_SRC1 () {
        s_texSRC1.filterMode = cudaFilterModePoint;
        cudaUnbindTexture(s_texSRC1);
    }

    __device__ T operator()(float x, float y) const { 
        return tex2D(s_texSRC1, s_.x * x, s_.x * y); 
    } 
};

template<typename T> 
struct gpu_sampler_SRC4 {
    __host__ gpu_sampler_SRC4(const gpu_image<T>& img, cudaTextureFilterMode filter_mode=cudaFilterModePoint) {
        s_texSRC4.filterMode = filter_mode;
        GPU_SAFE_CALL(cudaBindTexture2D(0, s_texSRC4, img.ptr(), img.w(), img.h(), img.pitch()));
    }

    __host__ ~gpu_sampler_SRC4() {
        s_texSRC4.filterMode = cudaFilterModePoint;
        cudaUnbindTexture(s_texSRC4);
    }
    
    __device__ T operator()(float x, float y) const { 
        return tex2D(s_texSRC4, x, y); 
    } 
};


template<typename T> 
struct gpu_resampler_SRC4 {
    float2 s_;

    __host__ gpu_resampler_SRC4(const gpu_image<T>& img, float2 s, cudaTextureFilterMode filter_mode=cudaFilterModePoint) {
        s_ = s;
        s_texSRC4.filterMode = filter_mode;
        GPU_SAFE_CALL(cudaBindTexture2D(0, s_texSRC4, img.ptr(), img.w(), img.h(), img.pitch()));
    }

    __host__ ~gpu_resampler_SRC4() {
        s_texSRC4.filterMode = cudaFilterModePoint;
        cudaUnbindTexture(s_texSRC4);
    }

    __device__ T operator()(float x, float y) const { 
        return tex2D(s_texSRC4, s_.x * x, s_.x * y); 
    } 
};

template<typename T> 
struct gpu_sampler_ST {
    __host__ gpu_sampler_ST(const gpu_image<T>& img, cudaTextureFilterMode filter_mode=cudaFilterModePoint) {
        s_texST.filterMode = filter_mode;
        GPU_SAFE_CALL(cudaBindTexture2D(0, s_texST, img.ptr(), img.w(), img.h(), img.pitch()));
    }

    __host__ ~gpu_sampler_ST() {
        s_texST.filterMode = cudaFilterModePoint;
        cudaUnbindTexture(s_texST);
    }
    
    __device__ T operator()(float x, float y) const { 
        return tex2D(s_texST, x, y); 
    } 
};


template<typename T> 
struct gpu_resampler_ST {
    float2 s_;

    __host__ gpu_resampler_ST(const gpu_image<T>& img, float2 s, cudaTextureFilterMode filter_mode=cudaFilterModePoint) {
        s_ = s;
        s_texST.filterMode = filter_mode;
        GPU_SAFE_CALL(cudaBindTexture2D(0, s_texST, img.ptr(), img.w(), img.h(), img.pitch()));
    }

    __host__ ~gpu_resampler_ST() {
        s_texST.filterMode = cudaFilterModePoint;
        cudaUnbindTexture(s_texST);
    }

    __device__ T operator()(float x, float y) const { 
        return tex2D(s_texST, s_.x * x, s_.x * y); 
    } 
};

template <typename T> 
struct gpu_constant_sampler {
    T value;

    gpu_constant_sampler(T v) : value(v) { }
    
    __host__ __device__ T operator()(float ix, float iy) const { 
        return value; 
    }
};

#endif

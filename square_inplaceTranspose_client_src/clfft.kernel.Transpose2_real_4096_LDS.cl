/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/

__attribute__(( reqd_work_group_size( 256, 1, 1 ) ))
kernel void
transpose_square( global float* restrict inputA )
{
   const int numGroupsY_1 = 8256; //(dim / 32) * (dim / 32 + 1) / 2
   size_t g_index;
   
   size_t iOffset = 0;
   g_index = get_group_id(0);
   iOffset += (g_index/numGroupsY_1)*16777216; // dim * dim
   g_index = g_index % numGroupsY_1;
   
   inputA += iOffset;
   global float *outputA = inputA;
   //m = dim/32 = 128
   //-257 = -2*m-1
   //66048 = 4*m*(m+1)
   float row = (-257+sqrt((66048-8.0f*g_index- 7)))/ (-2.0f);
   if (row == (float)(int)row) row -= 1; 
   const int t_gy = (int)row;
   
   const int t_gx_p = g_index - 128*t_gy + t_gy*(t_gy + 1) / 2;//m
   const int t_gy_p = t_gx_p - t_gy;
   
   const int d_lidx = get_local_id(0) % 16;
   const int d_lidy = get_local_id(0) / 16;
   
   const int lidy = (d_lidy * 16 + d_lidx) /32;
   const int lidx = (d_lidy * 16 + d_lidx) %32;
   
   const int idx = lidx + t_gx_p*32;
   const int idy = lidy + t_gy_p*32;
   
   const int starting_index_yx = t_gy_p*32 + t_gx_p*131072;//32 * dim
   
   __local float xy_s[1024+32];
   __local float yx_s[1024+32];
   float tmpm, tmpt;
   
   int index;
   for (int loop = 0; loop<4; ++loop){
      index = lidy*33 + lidx + loop*264;
      tmpm = inputA[(idy + loop *8)*4096 + idx];
      tmpt = inputA[(lidy + loop *8)*4096 + lidx + starting_index_yx];
      xy_s[index] = tmpm; 
      yx_s[index] = tmpt; 
   }
   
   barrier(CLK_LOCAL_MEM_FENCE);
   
   for (int loop = 0; loop<4; ++loop){
      index = lidx*33 + lidy + 8*loop;
      outputA[(idy + loop*8)*4096 + idx] = yx_s[index];
      outputA[(lidy + loop*8)*4096 + lidx+ starting_index_yx] = xy_s[index];
   }
}


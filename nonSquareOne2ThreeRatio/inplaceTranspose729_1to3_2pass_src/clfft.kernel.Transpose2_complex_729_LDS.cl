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
transpose_square( global float2* restrict inputA )
{
   //729 by 729 matrix
   const int numGroupsY_1 = 276;
   size_t g_index;
   
   size_t iOffset = 0;
   g_index = get_group_id(0);
   iOffset += (g_index/numGroupsY_1)*531441;//729*729
   g_index = g_index % numGroupsY_1;
   
   inputA += iOffset;
   global float2 *outputA = inputA;
   
   //m = dim/32 = 23
   //-47 = -2*m-1
   //2208 = 4*m*(m+1)
   float row = (-47+sqrt((2208-8.0f*g_index- 7)))/ (-2.0f);
   if (row == (float)(int)row) row -= 1; 
   const int t_gy = (int)row;
   
   const int t_gx_p = g_index - 23*t_gy + t_gy*(t_gy + 1) / 2;
   const int t_gy_p = t_gx_p - t_gy;
   
   const int d_lidx = get_local_id(0) % 16;
   const int d_lidy = get_local_id(0) / 16;
   
   const int lidy = (d_lidy * 16 + d_lidx) /32;
   const int lidx = (d_lidy * 16 + d_lidx) %32;
   
   const int idx = lidx + t_gx_p*32;
   const int idy = lidy + t_gy_p*32;
   
   const int starting_index_yx = t_gy_p*32 + t_gx_p*23328;//32*729
   
   __local float2 xy_s[1024];
   __local float2 yx_s[1024];
   float2 tmpm, tmpt;
   
   int index;
   
   if (729 - (t_gx_p + 1) *32>0){
      for (int loop = 0; loop<4; ++loop){
         index = lidy*32 + lidx + loop*256;
         tmpm = inputA[(idy + loop*8)*729 + idx];
         tmpt = inputA[(lidy + loop*8)*729 + lidx + starting_index_yx];
         xy_s[index] = tmpm;
         yx_s[index] = tmpt;
      }
   }
   else{
      for (int loop = 0; loop<4; ++loop){
         index = lidy*32 + lidx + loop*256;
         if ((idy + loop*8)<729&& idx<729)
            tmpm = inputA[(idy + loop*8)*729 + idx];
         if ((t_gy_p *32 + lidx)<729 && (t_gx_p * 32 + lidy + loop*8)<729) 
            tmpt = inputA[(lidy + loop*8)*729 + lidx + starting_index_yx];
         xy_s[index] = tmpm;
         yx_s[index] = tmpt;
         }
   }
   
   barrier(CLK_LOCAL_MEM_FENCE);
   
   if (729 - (t_gx_p + 1) *32>0){
      for (int loop = 0; loop<4; ++loop){
         index = lidx*32 + lidy + 8*loop ;
         outputA[(idy + loop*8)*729 + idx] = yx_s[index];
         outputA[(lidy + loop*8)*729 + lidx + starting_index_yx] = xy_s[index]; 
      }
   }
   else{
      for (int loop = 0; loop<4; ++loop){
         index = lidx*32 + lidy + 8*loop;
         if ((idy + loop*8)<729 && idx<729)
            outputA[(idy + loop*8)*729 + idx] = yx_s[index]; 
         if ((t_gy_p * 32 + lidx)<729 && (t_gx_p * 32 + lidy + loop*8)<729)
            outputA[(lidy + loop*8)*729 + lidx + starting_index_yx] = xy_s[index];
      }
   }
   
}


/*
 * SMART: string matching algorithms research tool.
 * Copyright (C) 2012  Simone Faro and Thierry Lecroq
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 * 
 * contact the authors at: faro@dmi.unict.it, thierry.lecroq@univ-rouen.fr
 * download the tool at: http://www.dmi.unict.it/~faro/smart/
 *
 * This is an implementation of the Forward Semplified BNDM algorithm
 * in S. Faro and T. Lecroq. 
 * Efficient Variants of the Backward-Oracle-Matching Algorithm. 
 * Proceedings of the Prague Stringology Conference 2008, pp.146--160, Czech Technical University in Prague, Czech Republic, (2008).
 */
#include <cstring>
#include "include/define.h"

namespace fsbndm{
/*
 * Forward Semplified BNDM algorithm designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

int search_large(unsigned char *x, int m, unsigned char *y, int n) {
   unsigned int B[SIGMA], D, set;
   int i, j, pos, mMinus1, count, p_len, k,s;
   
   p_len = m;
   m = 30;

   /* Preprocessing */
   count = 0;
   mMinus1 = m - 1;
   set = 1;
   for(i=0; i<SIGMA; i++) B[i]=set;
   for (i = 0; i < m; ++i) B[x[i]] |= (1<<(m-i));

   /* Searching */
   if(!std::memcmp(x,y,m)){
      return FALSE;
   }
   j = m;
   while (j < n) {
      D = (B[y[j+1]]<<1) & B[y[j]];
      if (D != 0) {
         pos = j;
         while (D=(D<<1) & B[y[j-1]]) --j;
         j += mMinus1;
         if (j == pos) {
            k = m; s=j-mMinus1;
            while (k<p_len && x[k]==y[s+k]) k++;
            if (k==p_len){
               return TRUE;
            }
            ++j;
         }
      }
      else j+=m;
   }  
   return FALSE;
}

int search(unsigned char *x, int m, unsigned char *y, int n) {
   unsigned int B[SIGMA], D, set;
   int i, j, pos, mMinus1, count;
   
   if(m>31) return fsbndm::search_large(x,m,y,n);

   /* Preprocessing */
   count = 0;
   mMinus1 = m - 1;
   set = 1;
   for(i=0; i<SIGMA; i++) B[i]=set;
   for (i = 0; i < m; ++i) B[x[i]] |= (1<<(m-i));

   /* Searching */
   if(!std::memcmp(x,y,m)){
      return FALSE;
   }
   j = m;
   while (j < n) {
      D = (B[y[j+1]]<<1) & B[y[j]];
      if (D != 0) {
         pos = j;
         while (D=(D<<1) & B[y[j-1]]) --j;
         j += mMinus1;
         if (j == pos) {
            return TRUE;
            // ++j;
         }
      }
      else j+=m;
   }
   return count;
}
}
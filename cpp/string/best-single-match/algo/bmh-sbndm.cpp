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
 * This is an implementation of the Horspool algorithm with BNDM test
 * in J. Holub and B. Durian. 
 * Talk: Fast variants of bit parallel approach to suffix automata. 
 * The Second Haifa Annual International Stringology Research Workshop of the Israeli Science Foundation, (2005).
 */
#include <cstring>
#include "include/define.h"

namespace bmh_sbndm{
/*
 * Horspool algorithm with BNDM test designed for large patterns
 * The present implementation searches for prefixes of the pattern of length 32.
 * When an occurrence is found the algorithm tests for the whole occurrence of the pattern
 */

int search_large(unsigned char *x, int m, unsigned char *y, int n) {
   int i, j,k,s, count, hbc[SIGMA], shift, p_len;
   unsigned int B[SIGMA], D;

   p_len = m;
   m = 32;
   int diff = p_len-m;

   /* Preprocessing */
   for(i=0;i<SIGMA;i++)
      hbc[i]=m;
   for(i=0;i<m;i++)
      hbc[x[i]]=m-i-1;
   for (i=0; i<SIGMA; i++)  
      B[i] = 0;
   for (i=0; i<m; i++) 
      B[x[m-i-1]] |= (unsigned int)1 << (i+WORD-m);
   for(i=0; i<m; i++) 
      y[n+i]=x[i];
   D = B[x[m-1]]; j=1; shift=1;
   for(i=m-2; i>0; i--, j++) {
      if(D & (1<<(m-1))) shift = j;
      D = (D<<1) & B[x[i]];
   }

   /* Searching */      
   count = 0;
   if(!std::memcmp(x,y,m)){
      return FALSE;
   }
   i = m;
   while(i+diff < n) {
      while( (k=hbc[y[i]])!=0 ) i+=k;
      j=i; s=i-m+1;
      D = B[y[j]];
      while(D!=0) { 
         j--;
         D = (D<<1) & B[y[j]];
      }
      if(j<s) {
         if(s<n) {
            k = m;
            while(k<p_len && x[k]==y[s+k]) k++;
            if(k==p_len && i+diff<n){
               return TRUE;
            }
         }
         i += shift;
      }
      else i = j+m;
   }
   return FALSE;
}

int search(unsigned char *x, int m, unsigned char *y, int n) {
   int i, j,k,s, count, hbc[SIGMA], shift;
   unsigned int B[SIGMA], D;

   if (m>32) return bmh_sbndm::search_large(x,m,y,n);   

   /* Preprocessing */
   for(i=0;i<SIGMA;i++) 
      hbc[i]=m;
   for(i=0;i<m;i++) 
      hbc[x[i]]=m-i-1;
   for (i=0; i<SIGMA; i++)  
      B[i] = 0;
   for (i=0; i<m; i++) 
      B[x[m-i-1]] |= (unsigned int)1 << (i+WORD-m);
   for(i=0; i<m; i++) 
      y[n+i]=x[i];
   D = B[x[m-1]]; j=1; shift=1;
   for(i=m-2; i>0; i--, j++) {
      if(D & (1<<(m-1))) shift = j;
      D = (D<<1) & B[x[i]];
   }

   /* Searching */      
   count = 0;
   if(!std::memcmp(x,y,m)){
      return FALSE;
   }
   i = m;
   while(i < n) {
      while( (k=hbc[y[i]])!=0 ) i+=k;
      j=i; s=i-m+1;
      D = B[y[j]];
      while(D!=0) { 
         j--;
         D = (D<<1) & B[y[j]];
      }
      if(j<s) {
         if(s<n && i<n){
            return TRUE;
         }
         // i += shift;
      }
      else i = j+m;
   }  
   return FALSE;
}
}

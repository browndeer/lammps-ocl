/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Stephen Foiles (SNL), Murray Daw (SNL)
      David Richie (Brown Deer Technology) - OpenCL modifications
------------------------------------------------------------------------- */

/* DAR */

#define __AMD__

#if defined(__AMD__) || defined(__coprthr__)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#include "/usr/local/browndeer/include/stdcl.h"

__kernel void
neighbor_kern1(
	int nlocal,
	double xbboxhi, double ybboxhi, double zbboxhi,
	double xbboxlo, double ybboxlo, double zbboxlo,
	double bininvx, double bininvy, double bininvz,
	int mbinxlo, int mbinylo, int mbinzlo,
	int nbinx, int nbiny, int nbinz,
	int mbinx, int mbiny, int mbinz,
	double cutneighsq,
	int nstencil,
	__read_only image2d_t img_stencil,
	__global int* binhead,
	__global int* bins,
	__global double* x_data,
	__global int* numneigh,
	__global int* nndataoffset,
	__global int* nndata
)
{

	const sampler_t sampler0 
		= CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	int gti = get_global_id(0);

	if (gti<nlocal) {

		int i = gti + 0;

		double xtmp = x_data[i*3+0];
		double ytmp = x_data[i*3+1];
		double ztmp = x_data[i*3+2];

		int nptr0 = i*135;
		int nptr = nptr0;
		nndataoffset[gti] = nptr0;

		int xhi = (int)((xtmp-xbboxhi)*bininvx) - mbinxlo + nbinx;
		int xlo = (int)((xtmp-xbboxlo)*bininvx) - mbinxlo;
		int yhi = (int)((ytmp-ybboxhi)*bininvy) - mbinylo + nbiny;
		int ylo = (int)((ytmp-ybboxlo)*bininvy) - mbinylo;
		int zhi = (int)((ztmp-zbboxhi)*bininvz) - mbinzlo + nbinz;
		int zlo = (int)((ztmp-zbboxlo)*bininvz) - mbinzlo;

		int ix = ((xtmp >= xbboxhi)? xhi : xlo);
		ix = ((xtmp < xbboxlo)? ix-1 : ix);

		int iy = ((ytmp >= ybboxhi)? yhi : ylo);
		iy = ((ytmp < ybboxlo)? iy-1 : iy);

		int iz = ((ztmp >= zbboxhi)? zhi : zlo);
		iz = ((ztmp < zbboxlo)? iz-1 : iz);

		int ibin = (iz*mbiny*mbinx + iy*mbinx + ix);

		int k;
		for(k = 0; k < nstencil; k++) {

			int4 s = read_imagei(img_stencil, sampler0, 
				__builtin_vector_int2(k,0) );

			int j = binhead[ibin+s.x];

			while (j >= 0) {

				if (i != j) {

					double delx = xtmp - x_data[j*3+0];
					double dely = ytmp - x_data[j*3+1];
					double delz = ztmp - x_data[j*3+2];
					double rsq = delx*delx + dely*dely + delz*delz;

   				if (rsq <= cutneighsq) nndata[nptr++] = j;

				}
  
				j = bins[j];

 
			}

		}

		numneigh[gti] = nptr - nptr0;

	}

}


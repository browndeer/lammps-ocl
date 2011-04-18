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
   Contributing authors: Paul Crozier (SNL),
      David Richie (Brown Deer Technology) - OpenCL modifications
------------------------------------------------------------------------- */

/* DAR */

#if defined(__AMD__) || defined(__coprthr__)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif


__kernel void
pair_lj_cut_kern(

   int eflag,
   int inum,
   int ilist0,

	int nall,

   double cutsq,

   __global int* numneigh,
   __global int* nndataoffset,
   __global int* nndata,
   __global double* x_data,

	double lj1, double lj2, double lj3, double lj4, double offset,
	__read_only image2d_t special_lj,

   __global double* f_data,
   __global double* eatom,
   __global double* vtmp

)
{

   const sampler_t sampler0
      = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

   int gti = get_global_id(0);

   if (gti<inum) {

		int i = gti + ilist0;

		double xtmp = x_data[i*3+0];
      double ytmp = x_data[i*3+1];
      double ztmp = x_data[i*3+2];

      int offset = nndataoffset[gti];

      int jnum = numneigh[gti];

      double tmpx = 0.0;
      double tmpy = 0.0;
      double tmpz = 0.0;
      double tmpe = 0.0;
      double vtmp0 = 0.0;
      double vtmp1 = 0.0;
      double vtmp2 = 0.0;
      double vtmp3 = 0.0;
      double vtmp4 = 0.0;
      double vtmp5 = 0.0;

      int jj;
      for (jj = 0; jj < jnum; jj++) {

			int j = nndata[offset+jj];

			double factor_lj;

			if (j < nall) factor_lj = 1.0;
			else {
				int m = j/nall;
				float4 c = read_imagef(special_lj, sampler0, (int2)(0,m));
   			factor_lj = as_double(c.xy);
   			j %= nall;
      	}

			double delx = xtmp - x_data[j*3+0];
         double dely = ytmp - x_data[j*3+1];
         double delz = ztmp - x_data[j*3+2];
         double rsq = delx*delx + dely*dely + delz*delz;

			if (rsq < cutsq) {

				double r2inv = 1.0/rsq;
   			double r6inv = r2inv*r2inv*r2inv;
   			double forcelj = r6inv * (lj1*r6inv - lj2);
   			double fpair = factor_lj*forcelj*r2inv;

				tmpx += delx*fpair;
				tmpy += dely*fpair;
				tmpz += delz*fpair;

				if (eflag == 1) {

					double evdwl = r6inv*(lj3*r6inv-lj4) - offset;
					evdwl *= factor_lj;
					tmpe += evdwl;

               vtmp0 += 0.5*delx*delx*fpair;
               vtmp1 += 0.5*dely*dely*fpair;
               vtmp2 += 0.5*delz*delz*fpair;
               vtmp3 += 0.5*delx*dely*fpair;
               vtmp4 += 0.5*delx*delz*fpair;
               vtmp5 += 0.5*dely*delz*fpair;

				}

			}

		}

		f_data[i*3+0] = tmpx;
      f_data[i*3+1] = tmpy; 
      f_data[i*3+2] = tmpz;

		if (eflag == 1) {

         eatom[i] = tmpe;
         vtmp[i*6+0] = vtmp0;
         vtmp[i*6+1] = vtmp1;
         vtmp[i*6+2] = vtmp2;
         vtmp[i*6+3] = vtmp3;
         vtmp[i*6+4] = vtmp4;
         vtmp[i*6+5] = vtmp5;

      }

	}

}


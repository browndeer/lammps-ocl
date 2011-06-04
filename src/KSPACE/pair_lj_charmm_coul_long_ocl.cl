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

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

typedef union {int i; float f;} union_int_float_t;

#define RTABLE 	0
#define FTABLE 	1
#define CTABLE		2
#define ETABLE		3

__kernel void
pair_lj_cut_kern(

   int eflag,
   int inum,
   int ilist0,

	int nall,
	int ncoultablebits,
	int ncoulmask,
	int ncoulshiftbits,

   double cut_bothsq,
   double cut_coulsq,
   double tabinnersq,
   double cut_ljsq,
   double cut_lj_innersq,

	double g_ewald,
	double qqrd2e,
	double denom_lj,

   __global int* numneigh,
   __global int* nndataoffset,
   __global int* nndata,
   __global double* x_data,
   __global double* q_data,

	double lj1, 
	double lj2, 
	double lj3, 
	double lj4, 
	double offset,

	__read_only image2d_t special_coul,
	__read_only image2d_t special_lj,
	__read_only image2d_t table_img,

   __global double* f_data,
   __global double* eatom_lj,
   __global double* eatom_coul,
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
      double qtmp = q_data[i];

      int offset = nndataoffset[gti];

      int jnum = numneigh[gti];

      double tmpx = 0.0;
      double tmpy = 0.0;
      double tmpz = 0.0;
      double tmpe_coul = 0.0;
      double tmpe_lj = 0.0;
      double vtmp0 = 0.0;
      double vtmp1 = 0.0;
      double vtmp2 = 0.0;
      double vtmp3 = 0.0;
      double vtmp4 = 0.0;
      double vtmp5 = 0.0;

      int jj;
      for (jj = 0; jj < jnum; jj++) {

			int j = nndata[offset+jj];

			double factor_coul;
			double factor_lj;

			if (j < nall) {
				factor_coul = 1.0;
				factor_lj = 1.0;
			} else {
				int m = j/nall;
				float4 c = read_imagef(special_coul, sampler0, (int2)(0,m));
   			factor_coul = as_double(c.xy);
				c = read_imagef(special_lj, sampler0, (int2)(0,m));
   			factor_lj = as_double(c.xy);
   			j %= nall;
      	}

			double delx = xtmp - x_data[j*3+0];
         double dely = ytmp - x_data[j*3+1];
         double delz = ztmp - x_data[j*3+2];
         double rsq = delx*delx + dely*dely + delz*delz;

			if (rsq < cut_bothsq) {

				double r2inv = 1.0/rsq;

				double forcecoul;
				double forcelj;
				double prefactor;
				double erfc;
				double fraction;
				int itable;
	
				double r6inv;

				if (rsq < cut_coulsq) {
				
					if (!ncoultablebits || rsq <= tabinnersq) {

						double r = sqrt(rsq);
						double grij = g_ewald * r;
						double expm2 = exp(-grij*grij);
						double t = 1.0 / (1.0 + EWALD_P*grij);
						erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
						prefactor = qqrd2e * qtmp*q_data[j]/r;
						forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
	
						if (factor_coul < 1.0) {
							forcecoul -= (1.0-factor_coul)*prefactor;
						} else {
							union_int_float_t rsq_lookup;
							rsq_lookup.f = rsq;
							itable = rsq_lookup.i & ncoulmask;
							itable >>= ncoulshiftbits;
							float4 c 
								= read_imagef(table_img,sampler0,(int2)(RTABLE,itable));
							double rtable = as_double(c.xy);
							double drtable = as_double(c.zw);
							fraction = (rsq_lookup.f - rtable)*drtable;
							c = read_imagef(table_img,sampler0,(int2)(FTABLE,itable));
							double ftable = as_double(c.xy);
							double dftable = as_double(c.zw);
							double table = ftable + fraction*dftable;
							forcecoul = qtmp*q_data[j] * table;
							if (factor_coul < 1.0) {
								c = read_imagef(
									table_img,sampler0,(int2)(CTABLE,itable));
								double ctable = as_double(c.xy);
								double dctable = as_double(c.zw);
								table = ctable + fraction*dctable;
								prefactor = qtmp*q_data[j] * table;
								forcecoul -= (1.0-factor_coul)*prefactor;
							}
						}
	
					}
	
				} else forcecoul = 0.0;


				if (rsq < cut_ljsq) {

					r6inv = r2inv*r2inv*r2inv;
					forcelj = r6inv * (lj1*r6inv - lj2);

					if (rsq > cut_lj_innersq) {
						double switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
							(cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
						double switch2 = 12.0*rsq * (cut_ljsq-rsq) *
							(rsq-cut_lj_innersq) / denom_lj;
						double philj = r6inv*(lj3*r6inv-lj4);
						forcelj = forcelj*switch1 + philj*switch2;
					}

				} else forcelj = 0.0;

   			double fpair = (forcecoul + factor_lj*forcelj) * r2inv;

				tmpx += delx*fpair;
				tmpy += dely*fpair;
				tmpz += delz*fpair;

				if (eflag == 1) {
	
					if (rsq < cut_coulsq) {

						double ecoul;
	
						if (!ncoultablebits || rsq <= tabinnersq) {

							ecoul = prefactor*erfc;

						} else {

							float4 c 
								= read_imagef(table_img,sampler0,(int2)(ETABLE,itable));
							double etable = as_double(c.xy);
							double detable = as_double(c.zw);
							double table = etable + fraction*detable;
							ecoul = qtmp*q_data[j] * table;

						} ecoul = 0.0;

						if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;

						tmpe_coul += ecoul;
	
					}
	
					if (rsq < cut_ljsq) {
	
						double evdwl = r6inv*(lj3*r6inv-lj4);
						if (rsq > cut_lj_innersq) {
							double switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
								(cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
							evdwl *= switch1;
						}
						evdwl *= factor_lj;
						tmpe_lj += evdwl;
	
					}

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

         eatom_coul[i] = tmpe_coul;
         eatom_lj[i] = tmpe_lj;
         vtmp[i*6+0] = vtmp0;
         vtmp[i*6+1] = vtmp1;
         vtmp[i*6+2] = vtmp2;
         vtmp[i*6+3] = vtmp3;
         vtmp[i*6+4] = vtmp4;
         vtmp[i*6+5] = vtmp5;

      }

	}

}


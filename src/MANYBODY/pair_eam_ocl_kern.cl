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
pair_eam_kern1(
	int inum,
	int ilist0,
	int nr,
	double rdr,
	double cutforcesq,
	__global int* numneigh,
	__global int* nndataoffset,
	__global int* nndata,
	__global double* x_data,
	int nrhor,
	__read_only image2d_t rhor_spline23,
	__read_only image2d_t rhor_spline45,
	__read_only image2d_t rhor_spline67,
	__global double* rho
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

		double tmp = 0.0;

		int jj;
		for (jj = 0; jj < jnum; jj++) {

			int j = nndata[offset+jj];

			double delx = xtmp - x_data[j*3+0];
			double dely = ytmp - x_data[j*3+1];
			double delz = ztmp - x_data[j*3+2];
			double rsq = delx*delx + dely*dely + delz*delz;

			if (rsq < cutforcesq) {
			
				double p = sqrt(rsq)*rdr + 1.0;
				int m = convert_int(p);
				m = min(m,nr-1);
				float pf = convert_float(m);
				p -= (double)pf;
				p = (p<1.0)? p : 1.0;

				int mx = m%nrhor;
				int my = m/nrhor;

				float4 c = read_imagef(rhor_spline23, sampler0, 
					__builtin_vector_int2(mx,my) );
				double coeff3 = as_double(c.zw);

				c = read_imagef(rhor_spline45, sampler0, 
					__builtin_vector_int2(mx,my) );
				double coeff4 = as_double(c.xy);
				double coeff5 = as_double(c.zw);

				c = read_imagef(rhor_spline67, sampler0, 
					__builtin_vector_int2(mx,my) );
				double coeff6 = as_double(c.xy);

				tmp += ((coeff3*p + coeff4)*p + coeff5)*p + coeff6;

			}

		}

		rho[gti] = tmp;

	}

}


__kernel void
pair_eam_kern2(
	int eflag,
	int inum,
	int nrho,
	double rdrho,
	__global double* rho,
	int nfrho,
	__read_only image2d_t frho_spline01,
	__read_only image2d_t frho_spline23,
	__read_only image2d_t frho_spline45,
	__read_only image2d_t frho_spline67,
	__global double* fp,
	__global double* phiatom
)
{

	const sampler_t sampler0 
		= CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	int gti = get_global_id(0);

	if (gti<inum) {

		double p = rho[gti]*rdrho + 1.0;

		int m = convert_int((float)p);
		m = max(1,min(m,nrho-1));
		float pf = convert_float(m);
		p -= (double)pf;
		p = (p<1.0)? p : 1.0;

		int mx = m%nfrho;
		int my = m/nfrho;

		float4 c01 = read_imagef(frho_spline01, sampler0, 
			__builtin_vector_int2(mx,my) );
		float4 c23 = read_imagef(frho_spline23, sampler0, 
			__builtin_vector_int2(mx,my) );

		double coeff0 = as_double(c01.xy);
		double coeff1 = as_double(c01.zw);
		double coeff2 = as_double(c23.xy);

		fp[gti] = (coeff0*p + coeff1)*p + coeff2;

		if (eflag == 1) {

			float4 c45 = read_imagef(frho_spline45, sampler0, 
				__builtin_vector_int2(mx,my) );
			float4 c67 = read_imagef(frho_spline67, sampler0, 
				__builtin_vector_int2(mx,my) );

			double coeff3 = as_double(c23.zw);
			double coeff4 = as_double(c45.xy);
			double coeff5 = as_double(c45.zw);
			double coeff6 = as_double(c67.xy);

			phiatom[gti] = ((coeff3*p + coeff4)*p + coeff5)*p + coeff6;

		}

	}

}


__kernel void
pair_eam_kern3(
	int eflag,
	int inum,
	int ilist0,
	int nr,
	double rdr,
	double cutforcesq,
	__global int* numneigh,
	__global int* nndataoffset,
	__global int* nndata,
	__global double* x_data,
	__global double* fp,
	int nrhor,
	__read_only image2d_t rhor_spline01,
	__read_only image2d_t rhor_spline23,
	int nz2r,
	__read_only image2d_t z2r_spline01,
	__read_only image2d_t z2r_spline23,
	__read_only image2d_t z2r_spline45,
	__read_only image2d_t z2r_spline67,
	__global double* f_data,
	__global double* phiatom,
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
		double tmpphi=0.0;
		double vtmp0=0.0;
		double vtmp1=0.0;
		double vtmp2=0.0;
		double vtmp3=0.0;
		double vtmp4=0.0;
		double vtmp5=0.0;
	
		int jj;
		for (jj = 0; jj < jnum; jj++) {

			int j = nndata[offset+jj];

      	double delx = xtmp - x_data[j*3+0];
      	double dely = ytmp - x_data[j*3+1];
      	double delz = ztmp - x_data[j*3+2];
      	double rsq = delx*delx + dely*dely + delz*delz;

			
      	if (rsq < cutforcesq) {

				double r = sqrt(rsq);
				double p = r*rdr + 1.0;

				int m = convert_int((float)p);
				m = min(m,nr-1);
				float pf = convert_float(m);
				p -= (double)pf;
				p = (p<1.0)? p : 1.0;

				int mx = m%nrhor;
				int my = m/nrhor;

				float4 c = read_imagef(rhor_spline01, sampler0, 
					__builtin_vector_int2(mx,my) );
				double coeff0 = as_double(c.xy);
				double coeff1 = as_double(c.zw);

				c = read_imagef(rhor_spline23, sampler0, 
					__builtin_vector_int2(mx,my) );
				double coeff2 = as_double(c.xy);

   			double rhoip = (coeff0*p + coeff1)*p + coeff2;
				
				c = read_imagef(rhor_spline01, sampler0, 
					__builtin_vector_int2(mx,my) );
				coeff0 = as_double(c.xy);
				coeff1 = as_double(c.zw);

				c = read_imagef(rhor_spline23, sampler0, 
					__builtin_vector_int2(mx,my) );
				coeff2 = as_double(c.xy);

   			double rhojp = (coeff0*p + coeff1)*p + coeff2;
			
				mx = m%nz2r;
				my = m/nz2r;

				c = read_imagef(z2r_spline01, sampler0, 
					__builtin_vector_int2(mx,my) );
				coeff0 = as_double(c.xy);
				coeff1 = as_double(c.zw);

				c = read_imagef(z2r_spline23, sampler0, 
					__builtin_vector_int2(mx,my));
				coeff2 = as_double(c.xy);
				double coeff3 = as_double(c.zw);

				c = read_imagef(z2r_spline45, sampler0, 
					__builtin_vector_int2(mx,my) );
				double coeff4 = as_double(c.xy);
				double coeff5 = as_double(c.zw);

				c = read_imagef(z2r_spline67, sampler0, 
					__builtin_vector_int2(mx,my) );
				double coeff6 = as_double(c.xy);

   			double z2p = (coeff0*p + coeff1)*p + coeff2;
				double z2 = ((coeff3*p + coeff4)*p + coeff5)*p + coeff6;
	
				double recip = 1.0/r;
				double phi = z2*recip;
				double phip = z2p*recip - phi*recip;
				double psip = fp[i]*rhojp + fp[j]*rhoip + phip;
				double fpair = -psip*recip;

				tmpx += delx*fpair;
				tmpy += dely*fpair;
				tmpz += delz*fpair;

				if (eflag == 1) {
					tmpphi += phi;
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
			phiatom[i] = tmpphi;
			vtmp[i*6+0] = vtmp0;
			vtmp[i*6+1] = vtmp1;
			vtmp[i*6+2] = vtmp2;
			vtmp[i*6+3] = vtmp3;
			vtmp[i*6+4] = vtmp4;
			vtmp[i*6+5] = vtmp5;
		}

	}

}


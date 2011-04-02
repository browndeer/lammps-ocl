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
   Contributing authors: 
		David Richie (Brown Deer Technology) - OpenCL modifications
------------------------------------------------------------------------- */

/* DAR */

#ifdef PAIR_CLASS

PairStyle(eam/ocl,PairEAM_OCL)

#else

#ifndef LMP_PAIR_EAM_OCL_H
#define LMP_PAIR_EAM_OCL_H

#include "stdio.h"
#include "pair.h"

#include <stdcl.h>

namespace LAMMPS_NS {

class PairEAM_OCL : public Pair {
 public:
  double cutforcesq,cutmax;

  // per-atom arrays

  double *rho,*fp;

  // potentials as array data

  int nrho,nr;
  int nfrho,nrhor,nz2r;
  double **frho,**rhor,**z2r;
  int *type2frho,**type2rhor,**type2z2r;
  
  // potentials in spline form used for force computation

  double dr,rdr,drho,rdrho;
  double ***rhor_spline,***frho_spline,***z2r_spline;

	double* frho_spline01;
	double* frho_spline23;
	double* frho_spline45;
	double* frho_spline67;

	double* rhor_spline01;
	double* rhor_spline23;
	double* rhor_spline45;
	double* rhor_spline67;

	double* z2r_spline01;
	double* z2r_spline23;
	double* z2r_spline45;
	double* z2r_spline67;

  PairEAM_OCL(class LAMMPS *);
  virtual ~PairEAM_OCL();
  void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  double single(int, int, int, int, double, double, double, double &);

  int pack_comm(int, int *, double *, int, int *);
  void unpack_comm(int, int, double *);
  int pack_reverse_comm(int, int, double *);
  void unpack_reverse_comm(int, int *, double *);
  double memory_usage();
  void swap_eam(double *, double **);

 protected:
  int nmax;                   // allocated size of per-atom arrays

  // potentials as file data

  int *map;                   // which element each atom type maps to

  struct Funcfl {
    char *file;
    int nrho,nr;
    double drho,dr,cut,mass;
    double *frho,*rhor,*zr;
  };
  Funcfl *funcfl;
  int nfuncfl;

  struct Setfl {
    char **elements;
    int nelements,nrho,nr;
    double drho,dr,cut;
    double *mass;
    double **frho,**rhor,***z2r;
  };
  Setfl *setfl;

  struct Fs {
    char **elements;
    int nelements,nrho,nr;
    double drho,dr,cut;
    double *mass;
    double **frho,***rhor,***z2r;
  };
  Fs *fs;

  void allocate();
  void array2spline();
  void interpolate(int, double, double *, double **);
  void grab(FILE *, int, double *);

  virtual void read_file(char *);
  virtual void file2array();

	void* clh;
	cl_kernel krn1;
	cl_kernel krn2;
	cl_kernel krn2e;
	cl_kernel krn3;
	cl_kernel krn3e;

	double* phiatom;
	double* vtmp;

};

}

#endif
#endif

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
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "stdcl.h"
#include "pair_lj_charmm_coul_long_ocl.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "kspace.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

enum{GEOMETRIC,ARITHMETIC,SIXTHPOWER};   // same as in pair.cpp

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ---------------------------------------------------------------------- */

PairLJCharmmCoulLong_OCL::PairLJCharmmCoulLong_OCL(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 1;
  ftable = NULL;
  implicit = 0;
  mix_flag = ARITHMETIC;

  clh = clopen(OCL_CONTEXT,0,CLLD_NOW);
  krn = clsym(OCL_CONTEXT,clh,"pair_lj_charmm_coul_long_kern",CLLD_NOW);

  special_lj_ocl = 0;
  special_coul_ocl = 0;

  eatom = 0;
  vtmp = 0;

}

/* ---------------------------------------------------------------------- */

PairLJCharmmCoulLong_OCL::~PairLJCharmmCoulLong_OCL()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(eps14);
    memory->destroy(sigma14);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(lj14_1);
    memory->destroy(lj14_2);
    memory->destroy(lj14_3);
    memory->destroy(lj14_4);
  }
  if (ftable) free_tables();
}

/* ---------------------------------------------------------------------- */

extern int devnum;

void PairLJCharmmCoulLong_OCL::compute(int eflag, int vflag)
{
  int i,ii,inum;
  double evdwl,ecoul;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  double* x_data = x[0];
  double* f_data = f[0];
//  double* q_data = q[0];
  double* q_data = q;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  int* nndataoffset = list->nndataoffset;
  int* nndata = list->nndata;

  if (atom->nmax > nmax) {
    memory->sfree(eatom_coul);
    memory->sfree(eatom_lj);
    memory->sfree(vtmp);
    nmax = atom->nmax;
    eatom_coul =(double*)memory->smalloc(nmax*sizeof(double),"pair:eatom_coul");
    eatom_lj =(double*)memory->smalloc(nmax*sizeof(double),"pair:eatom_lj");
    vtmp = (double*)memory->smalloc(6*atom->nmax*sizeof(double),"pair:vtmp");
    clmattach(OCL_CONTEXT,eatom_coul);
    clmattach(OCL_CONTEXT,eatom_lj);
    clmattach(OCL_CONTEXT,vtmp);
  }

  clndrange_t ndr = clndrange_init1d( 0, inum+WGSIZE1-inum%WGSIZE1, WGSIZE1 );

  int ilist0 = 0;

  double tmp_lj1 = lj1[1][1];
  double tmp_lj2 = lj2[1][1];
  double tmp_lj3 = lj3[1][1];
  double tmp_lj4 = lj4[1][1];

  clmsync(OCL_CONTEXT,devnum,x_data,CL_MEM_DEVICE|CL_EVENT_NOWAIT);
  clmsync(OCL_CONTEXT,devnum,q_data,CL_MEM_DEVICE|CL_EVENT_NOWAIT);

/* XXX WORKING HERE ON THE ARGS XXX */
  clarg_set(OCL_CONTEXT,krn,0,eflag);
  clarg_set(OCL_CONTEXT,krn,1,inum);
  clarg_set(OCL_CONTEXT,krn,2,ilist0);
  clarg_set(OCL_CONTEXT,krn,3,nall);
  clarg_set(OCL_CONTEXT,krn,4,ncoultablebits);
  clarg_set(OCL_CONTEXT,krn,5,ncoulmask);
  clarg_set(OCL_CONTEXT,krn,6,ncoulshiftbits);

  clarg_set(OCL_CONTEXT,krn,7,cut_bothsq);
  clarg_set(OCL_CONTEXT,krn,8,cut_coulsq);
  clarg_set(OCL_CONTEXT,krn,9,tabinnersq);
  clarg_set(OCL_CONTEXT,krn,10,cut_ljsq);
  clarg_set(OCL_CONTEXT,krn,11,cut_lj_innersq);

  clarg_set(OCL_CONTEXT,krn,12,g_ewald);
  clarg_set(OCL_CONTEXT,krn,13,qqrd2e);
  clarg_set(OCL_CONTEXT,krn,14,denom_lj);

  clarg_set_global(OCL_CONTEXT,krn,15,numneigh);
  clarg_set_global(OCL_CONTEXT,krn,16,nndataoffset);
  clarg_set_global(OCL_CONTEXT,krn,17,nndata);
  clarg_set_global(OCL_CONTEXT,krn,18,x_data);
  clarg_set_global(OCL_CONTEXT,krn,19,q_data);

  clarg_set(OCL_CONTEXT,krn,20,tmp_lj1);
  clarg_set(OCL_CONTEXT,krn,21,tmp_lj2);
  clarg_set(OCL_CONTEXT,krn,22,tmp_lj3);
  clarg_set(OCL_CONTEXT,krn,23,tmp_lj4);
  clarg_set(OCL_CONTEXT,krn,24,tmp_offset);

  clarg_set_global(OCL_CONTEXT,krn,25,special_coul_ocl);
  clarg_set_global(OCL_CONTEXT,krn,26,special_lj_ocl);
  clarg_set_global(OCL_CONTEXT,krn,27,table_img);

  clarg_set_global(OCL_CONTEXT,krn,28,f_data);
  clarg_set_global(OCL_CONTEXT,krn,29,eatom_coul);
  clarg_set_global(OCL_CONTEXT,krn,30,eatom_lj);
  clarg_set_global(OCL_CONTEXT,krn,31,vtmp);

  clfork(OCL_CONTEXT,devnum,krn,&ndr,CL_EVENT_NOWAIT);

  clmsync(OCL_CONTEXT,devnum,f_data,CL_MEM_HOST|CL_EVENT_NOWAIT);

  if (eflag || evflag) {

      clmsync(OCL_CONTEXT,devnum,eatom_coul,CL_MEM_HOST|CL_EVENT_NOWAIT);
      clmsync(OCL_CONTEXT,devnum,eatom_lj,CL_MEM_HOST|CL_EVENT_NOWAIT);
      clmsync(OCL_CONTEXT,devnum,vtmp,CL_MEM_HOST|CL_EVENT_NOWAIT);

      clwait(OCL_CONTEXT,devnum,CL_ALL_EVENT|CL_EVENT_RELEASE);

      if (eflag && eflag_either) {
         for (ii = 0; ii < inum; ii++) {
            i = ilist[ii];
            if (eflag_global) eng_vdwl += 0.5*eatom_lj[i];
            if (eflag_global) eng_coul += 0.5*eatom_coul[i];
            if (eflag_atom) eatom[i] += 0.5*(eatom_lj[i] + eatom_coul[i]);
         }
      }

      if (vflag_either) {
         for (ii = 0; ii < inum; ii++) {
            i = ilist[ii];
            if (vflag_global) {
               virial[0] += vtmp[i*6+0];
               virial[1] += vtmp[i*6+1];
               virial[2] += vtmp[i*6+2];
               virial[3] += vtmp[i*6+3];
               virial[4] += vtmp[i*6+4];
               virial[5] += vtmp[i*6+5];
            }
            if (vflag_atom) {
               vatom[i][0] += vtmp[i*6+0];
               vatom[i][1] += vtmp[i*6+1];
               vatom[i][2] += vtmp[i*6+2];
               vatom[i][3] += vtmp[i*6+3];
               vatom[i][4] += vtmp[i*6+4];
               vatom[i][5] += vtmp[i*6+5];
            }
         }
      }

   } else {

      clwait(OCL_CONTEXT,devnum,CL_ALL_EVENT|CL_EVENT_RELEASE);

   }

#if(0) 
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      if (j < nall) factor_coul = factor_lj = 1.0;
      else {
	factor_coul = special_coul[j/nall];
	factor_lj = special_lj[j/nall];
	j %= nall;
      }

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_bothsq) {
	r2inv = 1.0/rsq;

	if (rsq < cut_coulsq) {
	  if (!ncoultablebits || rsq <= tabinnersq) {
	    r = sqrt(rsq);
	    grij = g_ewald * r;
	    expm2 = exp(-grij*grij);
	    t = 1.0 / (1.0 + EWALD_P*grij);
	    erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
	    prefactor = qqrd2e * qtmp*q[j]/r;
	    forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
	    if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
	  } else {
	    union_int_float_t rsq_lookup;
	    rsq_lookup.f = rsq;
	    itable = rsq_lookup.i & ncoulmask;
	    itable >>= ncoulshiftbits;
	    fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
	    table = ftable[itable] + fraction*dftable[itable];
	    forcecoul = qtmp*q[j] * table;
	    if (factor_coul < 1.0) {
	      table = ctable[itable] + fraction*dctable[itable];
	      prefactor = qtmp*q[j] * table;
	      forcecoul -= (1.0-factor_coul)*prefactor;
	    }
	  }
	} else forcecoul = 0.0;

	if (rsq < cut_ljsq) {
	  r6inv = r2inv*r2inv*r2inv;
	  jtype = type[j];
	  forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
	  if (rsq > cut_lj_innersq) {
	    switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
	      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
	    switch2 = 12.0*rsq * (cut_ljsq-rsq) * 
	      (rsq-cut_lj_innersq) / denom_lj;
	    philj = r6inv * (lj3[itype][jtype]*r6inv - lj4[itype][jtype]);
	    forcelj = forcelj*switch1 + philj*switch2;
	  }
	} else forcelj = 0.0;

	fpair = (forcecoul + factor_lj*forcelj) * r2inv;

	f[i][0] += delx*fpair;
	f[i][1] += dely*fpair;
	f[i][2] += delz*fpair;
	if (newton_pair || j < nlocal) {
	  f[j][0] -= delx*fpair;
	  f[j][1] -= dely*fpair;
	  f[j][2] -= delz*fpair;
	}

	if (eflag) {
	  if (rsq < cut_coulsq) {
	    if (!ncoultablebits || rsq <= tabinnersq)
	      ecoul = prefactor*erfc;
	    else {
	      table = etable[itable] + fraction*detable[itable];
	      ecoul = qtmp*q[j] * table;
	    }
	    if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
	  } else ecoul = 0.0;

	  if (rsq < cut_ljsq) {
	    evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]);
	    if (rsq > cut_lj_innersq) {
	      switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
		(cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
	      evdwl *= switch1;
	    }
	    evdwl *= factor_lj;
	  } else evdwl = 0.0;
	}

	if (evflag) ev_tally(i,j,nlocal,newton_pair,
			     evdwl,ecoul,fpair,delx,dely,delz);
      }
    }
  }
#endif

  if (vflag_fdotr) virial_compute();
}

/* ---------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::compute_inner()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = listinner->inum;
  ilist = listinner->ilist;
  numneigh = listinner->numneigh;
  firstneigh = listinner->firstneigh;

  double cut_out_on = cut_respa[0];
  double cut_out_off = cut_respa[1];
  
  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      if (j < nall) factor_coul = factor_lj = 1.0;
      else {
	factor_coul = special_coul[j/nall];
	factor_lj = special_lj[j/nall];
	j %= nall;
      }

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq) {
        r2inv = 1.0/rsq;
        forcecoul = qqrd2e * qtmp*q[j]*sqrt(r2inv);
        if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*forcecoul;

	r6inv = r2inv*r2inv*r2inv;
	jtype = type[j];
	forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);

	fpair = (forcecoul + factor_lj*forcelj) * r2inv;

        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff; 
	  fpair *= 1.0 + rsw*rsw*(2.0*rsw-3.0);
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::compute_middle()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,fpair;
  double rsq,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double philj,switch1,switch2;
  double rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = listmiddle->inum;
  ilist = listmiddle->ilist;
  numneigh = listmiddle->numneigh;
  firstneigh = listmiddle->firstneigh;

  double cut_in_off = cut_respa[0];
  double cut_in_on = cut_respa[1];
  double cut_out_on = cut_respa[2];
  double cut_out_off = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;
  double cut_out_on_sq = cut_out_on*cut_out_on;
  double cut_out_off_sq = cut_out_off*cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];

      if (j < nall) factor_coul = factor_lj = 1.0;
      else {
	factor_coul = special_coul[j/nall];
	factor_lj = special_lj[j/nall];
	j %= nall;
      }
      
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_out_off_sq && rsq > cut_in_off_sq) {
	r2inv = 1.0/rsq;
        forcecoul = qqrd2e * qtmp*q[j]*sqrt(r2inv);
        if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*forcecoul;

	r6inv = r2inv*r2inv*r2inv;
	jtype = type[j];
	forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
	if (rsq > cut_lj_innersq) {
	  switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
	    (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
	  switch2 = 12.0*rsq * (cut_ljsq-rsq) * 
	    (rsq-cut_lj_innersq) / denom_lj;
	  philj = r6inv * (lj3[itype][jtype]*r6inv - lj4[itype][jtype]);
	  forcelj = forcelj*switch1 + philj*switch2;
	}

	fpair = (forcecoul + factor_lj*forcelj) * r2inv;
        if (rsq < cut_in_on_sq) {
	  rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff; 
	  fpair *= rsw*rsw*(3.0 - 2.0*rsw);
	}
        if (rsq > cut_out_on_sq) {
	  rsw = (sqrt(rsq) - cut_out_on)/cut_out_diff; 
	  fpair *= 1.0 + rsw*rsw*(2.0*rsw - 3.0);
	}

	f[i][0] += delx*fpair;
	f[i][1] += dely*fpair;
	f[i][2] += delz*fpair;
	if (newton_pair || j < nlocal) {
	  f[j][0] -= delx*fpair;
	  f[j][1] -= dely*fpair;
	  f[j][2] -= delz*fpair;
	}
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::compute_outer(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  double philj,switch1,switch2;
  double rsw;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;
  
  inum = listouter->inum;
  ilist = listouter->ilist;
  numneigh = listouter->numneigh;
  firstneigh = listouter->firstneigh;

  double cut_in_off = cut_respa[2];
  double cut_in_on = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_in_off_sq = cut_in_off*cut_in_off;
  double cut_in_on_sq = cut_in_on*cut_in_on;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      
      if (j < nall) factor_coul = factor_lj = 1.0;
      else {
	factor_coul = special_coul[j/nall];
	factor_lj = special_lj[j/nall];
	j %= nall;
      }
      
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      
      if (rsq < cut_bothsq) {
	r2inv = 1.0/rsq;
	
	if (rsq < cut_coulsq) {
	  if (!ncoultablebits || rsq <= tabinnersq) {
	    r = sqrt(rsq);
	    grij = g_ewald * r;
	    expm2 = exp(-grij*grij);
	    t = 1.0 / (1.0 + EWALD_P*grij);
	    erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
	    prefactor = qqrd2e * qtmp*q[j]/r;
	    forcecoul = prefactor * (erfc + EWALD_F*grij*expm2 - 1.0);
	    if (rsq > cut_in_off_sq) {
	      if (rsq < cut_in_on_sq) {
		rsw = (r - cut_in_off)/cut_in_diff; 
		forcecoul += prefactor*rsw*rsw*(3.0 - 2.0*rsw);
		if (factor_coul < 1.0)
		  forcecoul -= 
		    (1.0-factor_coul)*prefactor*rsw*rsw*(3.0 - 2.0*rsw);
	      } else {
		forcecoul += prefactor;
		if (factor_coul < 1.0)
		  forcecoul -= (1.0-factor_coul)*prefactor;
	      }
	    }
	  } else {
	    union_int_float_t rsq_lookup;
	    rsq_lookup.f = rsq;
	    itable = rsq_lookup.i & ncoulmask;
	    itable >>= ncoulshiftbits;
	    fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
	    table = ftable[itable] + fraction*dftable[itable];
	    forcecoul = qtmp*q[j] * table;
	    if (factor_coul < 1.0) {
	      table = ctable[itable] + fraction*dctable[itable];
	      prefactor = qtmp*q[j] * table;
	      forcecoul -= (1.0-factor_coul)*prefactor;
	    }
	  }
	} else forcecoul = 0.0;
	
	if (rsq < cut_ljsq && rsq > cut_in_off_sq) {
	  r6inv = r2inv*r2inv*r2inv;
	  forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
	  if (rsq > cut_lj_innersq) {
	    switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
	      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
	    switch2 = 12.0*rsq * (cut_ljsq-rsq) * 
	      (rsq-cut_lj_innersq) / denom_lj;
	    philj = r6inv * (lj3[itype][jtype]*r6inv - lj4[itype][jtype]);
	    forcelj = forcelj*switch1 + philj*switch2;
	  }
	  if (rsq < cut_in_on_sq) {
	    rsw = (sqrt(rsq) - cut_in_off)/cut_in_diff; 
	    forcelj *= rsw*rsw*(3.0 - 2.0*rsw);
	  }
	} else forcelj = 0.0;

	fpair = (forcecoul + forcelj) * r2inv;
	
	f[i][0] += delx*fpair;
	f[i][1] += dely*fpair;
	f[i][2] += delz*fpair;
	if (newton_pair || j < nlocal) {
	  f[j][0] -= delx*fpair;
	  f[j][1] -= dely*fpair;
	  f[j][2] -= delz*fpair;
	}
	
	if (eflag) {
	  if (rsq < cut_coulsq) {
	    if (!ncoultablebits || rsq <= tabinnersq) {
	      ecoul = prefactor*erfc;
	      if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
	    } else {
	      table = etable[itable] + fraction*detable[itable];
	      ecoul = qtmp*q[j] * table;
	      if (factor_coul < 1.0) {
		table = ptable[itable] + fraction*dptable[itable];
		prefactor = qtmp*q[j] * table;
		ecoul -= (1.0-factor_coul)*prefactor;
	      }
	    }
	  } else ecoul = 0.0;

	  if (rsq < cut_ljsq) {
	    r6inv = r2inv*r2inv*r2inv;
	    evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]);
	    if (rsq > cut_lj_innersq) {
	      switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
		(cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
	      evdwl *= switch1;
	    }
	    evdwl *= factor_lj;
	  } else evdwl = 0.0;
	}
	
	if (vflag) {
	  if (rsq < cut_coulsq) {
	    if (!ncoultablebits || rsq <= tabinnersq) {
	      forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
	      if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
	    } else {
	      table = vtable[itable] + fraction*dvtable[itable];
	      forcecoul = qtmp*q[j] * table;
	      if (factor_coul < 1.0) {
		table = ptable[itable] + fraction*dptable[itable];
		prefactor = qtmp*q[j] * table;
		forcecoul -= (1.0-factor_coul)*prefactor;
	      }
	    }
	  } else forcecoul = 0.0;

	  if (rsq <= cut_in_off_sq) {
	    r6inv = r2inv*r2inv*r2inv;
	    forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
	    if (rsq > cut_lj_innersq) {
	      switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
		(cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
	      switch2 = 12.0*rsq * (cut_ljsq-rsq) * 
		(rsq-cut_lj_innersq) / denom_lj;
	      philj = r6inv * (lj3[itype][jtype]*r6inv - lj4[itype][jtype]);
	      forcelj = forcelj*switch1 + philj*switch2;
	    }
	  } else if (rsq <= cut_in_on_sq) {
	    forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
	    if (rsq > cut_lj_innersq) {
	      switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
		(cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
	      switch2 = 12.0*rsq * (cut_ljsq-rsq) * 
		(rsq-cut_lj_innersq) / denom_lj;
	      philj = r6inv * (lj3[itype][jtype]*r6inv - lj4[itype][jtype]);
	      forcelj = forcelj*switch1 + philj*switch2;
	    }
	  }
	  
	  fpair = (forcecoul + factor_lj*forcelj) * r2inv;
	}

	if (evflag) ev_tally(i,j,nlocal,newton_pair,
			     evdwl,ecoul,fpair,delx,dely,delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(eps14,n+1,n+1,"pair:eps14");
  memory->create(sigma14,n+1,n+1,"pair:sigma14");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(lj14_1,n+1,n+1,"pair:lj14_1");
  memory->create(lj14_2,n+1,n+1,"pair:lj14_2");
  memory->create(lj14_3,n+1,n+1,"pair:lj14_3");
  memory->create(lj14_4,n+1,n+1,"pair:lj14_4");
}

/* ----------------------------------------------------------------------
   global settings
   unlike other pair styles,
     there are no individual pair settings that these override
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::settings(int narg, char **arg)
{
  if (narg != 2 && narg != 3) error->all("Illegal pair_style command");

  cut_lj_inner = force->numeric(arg[0]);
  cut_lj = force->numeric(arg[1]);
  if (narg == 2) cut_coul = cut_lj;
  else cut_coul = force->numeric(arg[2]);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::coeff(int narg, char **arg)
{
  if (narg != 4 && narg != 6) error->all("Illegal pair_coeff command");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(arg[2]);
  double sigma_one = force->numeric(arg[3]);
  double eps14_one = epsilon_one;
  double sigma14_one = sigma_one;
  if (narg == 6) {
    eps14_one = force->numeric(arg[4]);
    sigma14_one = force->numeric(arg[5]);
  }

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      eps14[i][j] = eps14_one;
      sigma14[i][j] = sigma14_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all("Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::init_style()
{
  if (!atom->q_flag)
    error->all("Pair style lj/charmm/coul/long requires atom attribute q");

  // request regular or rRESPA neighbor lists

  int irequest;

  if (update->whichflag == 1 && strcmp(update->integrate_style,"respa") == 0) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;

    if (respa == 0) irequest = neighbor->request(this);
    else if (respa == 1) {
      irequest = neighbor->request(this);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    } else {
      irequest = neighbor->request(this);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this);
      neighbor->requests[irequest]->id = 2;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respamiddle = 1;
      irequest = neighbor->request(this);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    }

  } else irequest = neighbor->request(this);

  // require cut_lj_inner < cut_lj

  if (cut_lj_inner >= cut_lj) 
    error->all("Pair inner cutoff >= Pair outer cutoff");

  cut_lj_innersq = cut_lj_inner * cut_lj_inner;
  cut_ljsq = cut_lj * cut_lj;
  cut_coulsq = cut_coul * cut_coul;
  cut_bothsq = MAX(cut_ljsq,cut_coulsq);

  denom_lj = (cut_ljsq-cut_lj_innersq) * (cut_ljsq-cut_lj_innersq) * 
    (cut_ljsq-cut_lj_innersq);

  // set & error check interior rRESPA cutoffs

  if (strcmp(update->integrate_style,"respa") == 0 &&
      ((Respa *) update->integrate)->level_inner >= 0) {
    cut_respa = ((Respa *) update->integrate)->cutoff;
    if (MIN(cut_lj,cut_coul) < cut_respa[3])
      error->all("Pair cutoff < Respa interior cutoff");
    if (cut_lj_inner < cut_respa[1])
      error->all("Pair inner cutoff < Respa interior cutoff");
  } else cut_respa = NULL;

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == NULL)
    error->all("Pair style is incompatible with KSpace style");
  g_ewald = force->kspace->g_ewald;
  
  // setup force tables

  if (ncoultablebits) init_tables();
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCharmmCoulLong_OCL::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
			       sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    eps14[i][j] = mix_energy(eps14[i][i],eps14[j][j],
			       sigma14[i][i],sigma14[j][j]);
    sigma14[i][j] = mix_distance(sigma14[i][i],sigma14[j][j]);
  }

  double cut = MAX(cut_lj,cut_coul);

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj14_1[i][j] = 48.0 * eps14[i][j] * pow(sigma14[i][j],12.0);
  lj14_2[i][j] = 24.0 * eps14[i][j] * pow(sigma14[i][j],6.0);
  lj14_3[i][j] = 4.0 * eps14[i][j] * pow(sigma14[i][j],12.0);
  lj14_4[i][j] = 4.0 * eps14[i][j] * pow(sigma14[i][j],6.0);
     
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  lj14_1[j][i] = lj14_1[i][j];
  lj14_2[j][i] = lj14_2[i][j];
  lj14_3[j][i] = lj14_3[i][j];
  lj14_4[j][i] = lj14_4[i][j];

  return cut;
}

/* ----------------------------------------------------------------------
   setup force tables used in compute routines
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::init_tables()
{
  int masklo,maskhi;
  double r,grij,expm2,derfc,rsw;
  double qqrd2e = force->qqrd2e;

  tabinnersq = tabinner*tabinner;
  init_bitmap(tabinner,cut_coul,ncoultablebits,
	      masklo,maskhi,ncoulmask,ncoulshiftbits);
  
  int ntable = 1;
  for (int i = 0; i < ncoultablebits; i++) ntable *= 2;
  
  // linear lookup tables of length N = 2^ncoultablebits
  // stored value = value at lower edge of bin
  // d values = delta from lower edge to upper edge of bin

  if (ftable) free_tables();
  
  memory->create(rtable,ntable,"pair:rtable");
  memory->create(ftable,ntable,"pair:ftable");
  memory->create(ctable,ntable,"pair:ctable");
  memory->create(etable,ntable,"pair:etable");
  memory->create(drtable,ntable,"pair:drtable");
  memory->create(dftable,ntable,"pair:dftable");
  memory->create(dctable,ntable,"pair:dctable");
  memory->create(detable,ntable,"pair:detable");

  if (cut_respa == NULL) {
    vtable = ptable = dvtable = dptable = NULL;
  } else {
    memory->create(vtable,ntable,"pair:vtable");
    memory->create(ptable,ntable,"pair:ptable");
    memory->create(dvtable,ntable,"pair:dvtable");
    memory->create(dptable,ntable,"pair:dptable");
  }

  union_int_float_t rsq_lookup;
  union_int_float_t minrsq_lookup;
  int itablemin;
  minrsq_lookup.i = 0 << ncoulshiftbits;
  minrsq_lookup.i |= maskhi;
    
  for (int i = 0; i < ntable; i++) {
    rsq_lookup.i = i << ncoulshiftbits;
    rsq_lookup.i |= masklo;
    if (rsq_lookup.f < tabinnersq) {
      rsq_lookup.i = i << ncoulshiftbits;
      rsq_lookup.i |= maskhi;
    }
    r = sqrtf(rsq_lookup.f);
    grij = g_ewald * r;
    expm2 = exp(-grij*grij);
    derfc = erfc(grij);
    if (cut_respa == NULL) {
      rtable[i] = rsq_lookup.f;
      ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      ctable[i] = qqrd2e/r;
      etable[i] = qqrd2e/r * derfc;
    } else {
      rtable[i] = rsq_lookup.f;
      ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2 - 1.0);
      ctable[i] = 0.0;
      etable[i] = qqrd2e/r * derfc;
      ptable[i] = qqrd2e/r;
      vtable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      if (rsq_lookup.f > cut_respa[2]*cut_respa[2]) {
	if (rsq_lookup.f < cut_respa[3]*cut_respa[3]) {
	  rsw = (r - cut_respa[2])/(cut_respa[3] - cut_respa[2]); 
	  ftable[i] += qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
	  ctable[i] = qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
	} else {
	  ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
	  ctable[i] = qqrd2e/r;
	}
      }
    }
    minrsq_lookup.f = MIN(minrsq_lookup.f,rsq_lookup.f);
  }

  tabinnersq = minrsq_lookup.f;
  
  int ntablem1 = ntable - 1;
  
  for (int i = 0; i < ntablem1; i++) {
    drtable[i] = 1.0/(rtable[i+1] - rtable[i]);
    dftable[i] = ftable[i+1] - ftable[i];
    dctable[i] = ctable[i+1] - ctable[i];
    detable[i] = etable[i+1] - etable[i];
  }

  if (cut_respa) {
    for (int i = 0; i < ntablem1; i++) {
      dvtable[i] = vtable[i+1] - vtable[i];
      dptable[i] = ptable[i+1] - ptable[i];
    }
  }
  
  // get the delta values for the last table entries 
  // tables are connected periodically between 0 and ntablem1
    
  drtable[ntablem1] = 1.0/(rtable[0] - rtable[ntablem1]);
  dftable[ntablem1] = ftable[0] - ftable[ntablem1];
  dctable[ntablem1] = ctable[0] - ctable[ntablem1];
  detable[ntablem1] = etable[0] - etable[ntablem1];
  if (cut_respa) {
    dvtable[ntablem1] = vtable[0] - vtable[ntablem1];
    dptable[ntablem1] = ptable[0] - ptable[ntablem1];
  }

  // get the correct delta values at itablemax    
  // smallest r is in bin itablemin
  // largest r is in bin itablemax, which is itablemin-1,
  //   or ntablem1 if itablemin=0
  // deltas at itablemax only needed if corresponding rsq < cut*cut
  // if so, compute deltas between rsq and cut*cut 
	
  double f_tmp,c_tmp,e_tmp,p_tmp,v_tmp;
  itablemin = minrsq_lookup.i & ncoulmask;
  itablemin >>= ncoulshiftbits;  
  int itablemax = itablemin - 1; 
  if (itablemin == 0) itablemax = ntablem1;     
  rsq_lookup.i = itablemax << ncoulshiftbits;
  rsq_lookup.i |= maskhi;

  if (rsq_lookup.f < cut_coulsq) {
    rsq_lookup.f = cut_coulsq;  
    r = sqrtf(rsq_lookup.f);
    grij = g_ewald * r;
    expm2 = exp(-grij*grij);
    derfc = erfc(grij);

    if (cut_respa == NULL) {
      f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      c_tmp = qqrd2e/r;
      e_tmp = qqrd2e/r * derfc;
    } else {
      f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2 - 1.0);
      c_tmp = 0.0;
      e_tmp = qqrd2e/r * derfc;
      p_tmp = qqrd2e/r;
      v_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      if (rsq_lookup.f > cut_respa[2]*cut_respa[2]) {
        if (rsq_lookup.f < cut_respa[3]*cut_respa[3]) {
          rsw = (r - cut_respa[2])/(cut_respa[3] - cut_respa[2]); 
          f_tmp += qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
          c_tmp = qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
        } else {
          f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
          c_tmp = qqrd2e/r;
        }
      }
    }

    drtable[itablemax] = 1.0/(rsq_lookup.f - rtable[itablemax]);   
    dftable[itablemax] = f_tmp - ftable[itablemax];
    dctable[itablemax] = c_tmp - ctable[itablemax];
    detable[itablemax] = e_tmp - etable[itablemax];
    if (cut_respa) {
      dvtable[itablemax] = v_tmp - vtable[itablemax];
      dptable[itablemax] = p_tmp - ptable[itablemax];
    }   
  }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
	fwrite(&epsilon[i][j],sizeof(double),1,fp);
	fwrite(&sigma[i][j],sizeof(double),1,fp);
	fwrite(&eps14[i][j],sizeof(double),1,fp);
	fwrite(&sigma14[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
	if (me == 0) {
	  fread(&epsilon[i][j],sizeof(double),1,fp);
	  fread(&sigma[i][j],sizeof(double),1,fp);
	  fread(&eps14[i][j],sizeof(double),1,fp);
	  fread(&sigma14[i][j],sizeof(double),1,fp);
	}
	MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&eps14[i][j],1,MPI_DOUBLE,0,world);
	MPI_Bcast(&sigma14[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::write_restart_settings(FILE *fp)
{
  fwrite(&cut_lj_inner,sizeof(double),1,fp);
  fwrite(&cut_lj,sizeof(double),1,fp);
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_lj_inner,sizeof(double),1,fp);
    fread(&cut_lj,sizeof(double),1,fp);
    fread(&cut_coul,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_lj_inner,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_lj,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   free memory for tables used in pair computations
------------------------------------------------------------------------- */

void PairLJCharmmCoulLong_OCL::free_tables()
{
  memory->destroy(rtable);
  memory->destroy(drtable);
  memory->destroy(ftable);
  memory->destroy(dftable);
  memory->destroy(ctable);
  memory->destroy(dctable);
  memory->destroy(etable);
  memory->destroy(detable);
  memory->destroy(vtable);
  memory->destroy(dvtable);
  memory->destroy(ptable);
  memory->destroy(dptable);
}

/* ---------------------------------------------------------------------- */

double PairLJCharmmCoulLong_OCL::single(int i, int j, int itype, int jtype,
				    double rsq,
				    double factor_coul, double factor_lj,
				    double &fforce)
{
  double r2inv,r6inv,r,grij,expm2,t,erfc,prefactor;
  double switch1,switch2,fraction,table,forcecoul,forcelj,phicoul,philj;
  int itable;

  r2inv = 1.0/rsq;
  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq) {
      r = sqrt(rsq);
      grij = g_ewald * r;
      expm2 = exp(-grij*grij);
      t = 1.0 / (1.0 + EWALD_P*grij);
      erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
      prefactor = force->qqrd2e * atom->q[i]*atom->q[j]/r;
      forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
      if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
    } else {
      union_int_float_t rsq_lookup;
      rsq_lookup.f = rsq;
      itable = rsq_lookup.i & ncoulmask;
      itable >>= ncoulshiftbits;
      fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
      table = ftable[itable] + fraction*dftable[itable];
      forcecoul = atom->q[i]*atom->q[j] * table;
      if (factor_coul < 1.0) {
	table = ctable[itable] + fraction*dctable[itable];
	prefactor = atom->q[i]*atom->q[j] * table;
	forcecoul -= (1.0-factor_coul)*prefactor;
      }
    }
  } else forcecoul = 0.0;
  if (rsq < cut_ljsq) {
    r6inv = r2inv*r2inv*r2inv;
    forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
    if (rsq > cut_lj_innersq) {
      switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
	(cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
      switch2 = 12.0*rsq * (cut_ljsq-rsq) * 
	(rsq-cut_lj_innersq) / denom_lj;
      philj = r6inv * (lj3[itype][jtype]*r6inv - lj4[itype][jtype]);
      forcelj = forcelj*switch1 + philj*switch2;
    }
  } else forcelj = 0.0;
  fforce = (forcecoul + factor_lj*forcelj) * r2inv;
  
  double eng = 0.0;
  if (rsq < cut_coulsq) {
    if (!ncoultablebits || rsq <= tabinnersq)
      phicoul = prefactor*erfc;
    else {
      table = etable[itable] + fraction*detable[itable];
      phicoul = atom->q[i]*atom->q[j] * table;
    }
    if (factor_coul < 1.0) phicoul -= (1.0-factor_coul)*prefactor;
    eng += phicoul;
  }

  if (rsq < cut_ljsq) {
    philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]);
    if (rsq > cut_lj_innersq) {
      switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
	(cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
      philj *= switch1;
    }
    eng += factor_lj*philj;
  }

  return eng;
}

/* ---------------------------------------------------------------------- */

void *PairLJCharmmCoulLong_OCL::extract(char *str, int &dim)
{
  dim = 2;
  if (strcmp(str,"lj14_1") == 0) return (void *) lj14_1;
  if (strcmp(str,"lj14_2") == 0) return (void *) lj14_2;
  if (strcmp(str,"lj14_3") == 0) return (void *) lj14_3;
  if (strcmp(str,"lj14_4") == 0) return (void *) lj14_4;

  dim = 0;
  if (strcmp(str,"implicit") == 0) return (void *) &implicit;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;

  return NULL;
}

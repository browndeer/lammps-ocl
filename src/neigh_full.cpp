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


#include "stdcl.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "atom.h"
#include "group.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   N^2 search for all neighbors
   every neighbor pair appears in list of both atoms i and j
------------------------------------------------------------------------- */

void Neighbor::full_nsq(NeighList *list)
{
  int i,j,n,itype,jtype,which,bitmask;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *neighptr;

  int **special = atom->special;
  int **nspecial = atom->nspecial;
  int *tag = atom->tag;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int molecular = atom->molecular;
  if (includegroup) {
    nlocal = atom->nfirst;
    bitmask = group->bitmask[includegroup];
  }

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int **pages = list->pages;

  int inum = 0;
  int npage = 0;
  int npnt = 0;

  // loop over owned atoms, storing neighbors

  for (i = 0; i < nlocal; i++) {

    if (pgsize - npnt < oneatom) {
      npnt = 0;
      npage++;
      if (npage == list->maxpage) pages = list->add_pages();
    }

    neighptr = &pages[npage][npnt];
    n = 0;

    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop over all atoms, owned and ghost
    // skip i = j

    for (j = 0; j < nall; j++) {
      if (includegroup && !(mask[j] & bitmask)) continue;
      if (i == j) continue;
      jtype = type[j];
      if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq <= cutneighsq[itype][jtype]) {
	if (molecular) which = find_special(special[i],nspecial[i],tag[j]);
	else which = 0;
	if (which == 0) neighptr[n++] = j;
	else if (which > 0) neighptr[n++] = which*nall + j;
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    npnt += n;
    if (n > oneatom || npnt >= pgsize)
      error->one("Neighbor list overflow, boost neigh_modify one or page");
  }

  list->inum = inum;
  list->gnum = 0;
}

/* ----------------------------------------------------------------------
   N^2 search for all neighbors
   include neighbors of ghost atoms
   every neighbor pair appears in list of both atoms i and j
------------------------------------------------------------------------- */

void Neighbor::full_nsq_ghost(NeighList *list)
{
  int i,j,n,itype,jtype,which;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *neighptr;

  int **special = atom->special;
  int **nspecial = atom->nspecial;
  int *tag = atom->tag;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int molecular = atom->molecular;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int **pages = list->pages;

  int inum = 0;
  int npage = 0;
  int npnt = 0;

  // loop over owned & ghost atoms, storing neighbors

  for (i = 0; i < nall; i++) {

    if (pgsize - npnt < oneatom) {
      npnt = 0;
      npage++;
      if (npage == list->maxpage) pages = list->add_pages();
    }

    neighptr = &pages[npage][npnt];
    n = 0;

    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop over all atoms, owned and ghost
    // skip i = j

    if (i < nlocal) {
      for (j = 0; j < nall; j++) {
	if (i == j) continue;
	jtype = type[j];
	if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;
	
	delx = xtmp - x[j][0];
	dely = ytmp - x[j][1];
	delz = ztmp - x[j][2];
	rsq = delx*delx + dely*dely + delz*delz;
	if (rsq <= cutneighsq[itype][jtype]) {
	  if (molecular) which = find_special(special[i],nspecial[i],tag[j]);
	  else which = 0;
	  if (which == 0) neighptr[n++] = j;
	  else if (which > 0) neighptr[n++] = which*nall + j;
	}
      }
    } else {
      for (j = 0; j < nall; j++) {
	if (i == j) continue;
	jtype = type[j];
	if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;
	
	delx = xtmp - x[j][0];
	dely = ytmp - x[j][1];
	delz = ztmp - x[j][2];
	rsq = delx*delx + dely*dely + delz*delz;
	if (rsq <= cutneighghostsq[itype][jtype]) {
	  if (molecular) which = find_special(special[i],nspecial[i],tag[j]);
	  else which = 0;
	  if (which == 0) neighptr[n++] = j;
	  else if (which > 0) neighptr[n++] = which*nall + j;
	}
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    npnt += n;
    if (n > oneatom || npnt >= pgsize)
      error->one("Neighbor list overflow, boost neigh_modify one or page");
  }

  list->inum = atom->nlocal;
  list->gnum = inum - atom->nlocal;
}

/* ----------------------------------------------------------------------
   binned neighbor list construction for all neighbors
   every neighbor pair appears in list of both atoms i and j
------------------------------------------------------------------------- */

void Neighbor::full_bin(NeighList *list)
{
  int i,j,k,n,itype,jtype,ibin,which;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *neighptr;

  // bin owned & ghost atoms

  bin_atoms();

  int **special = atom->special;
  int **nspecial = atom->nspecial;
  int *tag = atom->tag;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int molecular = atom->molecular;
  if (includegroup) nlocal = atom->nfirst;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int **pages = list->pages;
  int nstencil = list->nstencil;
  int *stencil = list->stencil;

  int inum = 0;
  int npage = 0;
  int npnt = 0;

  // loop over owned atoms, storing neighbors

  for (i = 0; i < nlocal; i++) {

    if (pgsize - npnt < oneatom) {
      npnt = 0;
      npage++;
      if (npage == list->maxpage) pages = list->add_pages();
    }

    neighptr = &pages[npage][npnt];
    n = 0;

    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop over all atoms in surrounding bins in stencil including self
    // skip i = j

    ibin = coord2bin(x[i]);

    for (k = 0; k < nstencil; k++) {
      for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
	if (i == j) continue;

	jtype = type[j];
	if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

	delx = xtmp - x[j][0];
	dely = ytmp - x[j][1];
	delz = ztmp - x[j][2];
	rsq = delx*delx + dely*dely + delz*delz;

	if (rsq <= cutneighsq[itype][jtype]) {
	  if (molecular) which = find_special(special[i],nspecial[i],tag[j]);
	  else which = 0;
	  if (which == 0) neighptr[n++] = j;
	  else if (which > 0) neighptr[n++] = which*nall + j;
	}
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    npnt += n;
    if (n > oneatom || npnt >= pgsize)
      error->one("Neighbor list overflow, boost neigh_modify one or page");
  }

  list->inum = inum;
  list->gnum = 0;
}

extern int devnum;

void Neighbor::full_bin_ocl(NeighList *list)
{
  int i,j,k,n,itype,jtype,ibin,which;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *neighptr;

  // bin local & ghost atoms

  bin_atoms();

	clmsync(OCL_CONTEXT,devnum,binhead,CL_MEM_DEVICE|CL_EVENT_NOWAIT);
	clmsync(OCL_CONTEXT,devnum,bins,CL_MEM_DEVICE|CL_EVENT_NOWAIT);


  // loop over each atom, storing neighbors

  int **special = atom->special;
  int **nspecial = atom->nspecial;
  int *tag = atom->tag;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int molecular = atom->molecular;
  if (includegroup) nlocal = atom->nfirst;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int **pages = list->pages;
  int nstencil = list->nstencil;
  int *stencil = list->stencil;

  int inum = 0;
  int npage = 0;
  int npnt = 0;


	if (!list->nndataoffset) { 
		list->nndataoffset = (int*)
         clmalloc(OCL_CONTEXT,nlocal*sizeof(int),0);
	}

	if (!list->nndata) {
		list->nndata = (int*)clmalloc(OCL_CONTEXT,nlocal*135*sizeof(int),0);
	}


	int* nndataoffset = list->nndataoffset;
	int* nndata = list->nndata;

	static int* img_stencil = 0;

	if (img_stencil == 0) {

		img_stencil 
			= (int*)clmalloc(OCL_CONTEXT,4*nstencil*sizeof(int),CL_MEM_DETACHED);
		cl_image_format fmt;
		fmt.image_channel_order = CL_RGBA;
		fmt.image_channel_data_type = CL_SIGNED_INT32;
		clmctl(img_stencil,CL_MCTL_SET_IMAGE2D,4*nstencil,1,&fmt);
  	 	clmattach(OCL_CONTEXT,img_stencil);
	
		for (k = 0; k < nstencil; k++) {
			img_stencil[4*k+0] = stencil[k];
		}

		clmsync(OCL_CONTEXT,devnum,img_stencil,CL_MEM_DEVICE|CL_EVENT_NOWAIT);

	}

	double* x_data = x[0];

	clmsync(OCL_CONTEXT,devnum,x_data,CL_MEM_DEVICE|CL_EVENT_NOWAIT);

	clarg_set(OCL_CONTEXT,krn1,0,nlocal);
	clarg_set(OCL_CONTEXT,krn1,1,bboxhi[0]);
	clarg_set(OCL_CONTEXT,krn1,2,bboxhi[1]);
	clarg_set(OCL_CONTEXT,krn1,3,bboxhi[2]);
	clarg_set(OCL_CONTEXT,krn1,4,bboxlo[0]);
	clarg_set(OCL_CONTEXT,krn1,5,bboxlo[1]);
	clarg_set(OCL_CONTEXT,krn1,6,bboxlo[2]);
	clarg_set(OCL_CONTEXT,krn1,7,bininvx);
	clarg_set(OCL_CONTEXT,krn1,8,bininvy);
	clarg_set(OCL_CONTEXT,krn1,9,bininvz);
	clarg_set(OCL_CONTEXT,krn1,10,mbinxlo);
	clarg_set(OCL_CONTEXT,krn1,11,mbinylo);
	clarg_set(OCL_CONTEXT,krn1,12,mbinzlo);
	clarg_set(OCL_CONTEXT,krn1,13,nbinx);
	clarg_set(OCL_CONTEXT,krn1,14,nbiny);
	clarg_set(OCL_CONTEXT,krn1,15,nbinz);
	clarg_set(OCL_CONTEXT,krn1,16,mbinx);
	clarg_set(OCL_CONTEXT,krn1,17,mbiny);
	clarg_set(OCL_CONTEXT,krn1,18,mbinz);
	clarg_set(OCL_CONTEXT,krn1,19,cutneighsq[1][1]);
	clarg_set(OCL_CONTEXT,krn1,20,nstencil);
	clarg_set_global(OCL_CONTEXT,krn1,21,img_stencil);
	clarg_set_global(OCL_CONTEXT,krn1,22,binhead);
	clarg_set_global(OCL_CONTEXT,krn1,23,bins);
	clarg_set_global(OCL_CONTEXT,krn1,24,x_data);
	clarg_set_global(OCL_CONTEXT,krn1,25,numneigh);
	clarg_set_global(OCL_CONTEXT,krn1,26,nndataoffset);
	clarg_set_global(OCL_CONTEXT,krn1,27,nndata);

	clndrange_t ndr = clndrange_init1d( 0, nlocal+WGSIZE-nlocal%WGSIZE, WGSIZE );

	clfork(OCL_CONTEXT,devnum,krn1,&ndr,CL_EVENT_NOWAIT);

	clwait(OCL_CONTEXT,devnum,CL_ALL_EVENT|CL_EVENT_RELEASE);

  list->inum = nlocal;

	for(int i=0;i<nlocal;i++) ilist[i] = i;

}

/* ----------------------------------------------------------------------
   binned neighbor list construction for all neighbors
   include neighbors of ghost atoms
   every neighbor pair appears in list of both atoms i and j
------------------------------------------------------------------------- */

void Neighbor::full_bin_ghost(NeighList *list)
{
  int i,j,k,n,itype,jtype,ibin,which;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int xbin,ybin,zbin,xbin2,ybin2,zbin2;
  int *neighptr;

  // bin owned & ghost atoms

  bin_atoms();

  int **special = atom->special;
  int **nspecial = atom->nspecial;
  int *tag = atom->tag;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int molecular = atom->molecular;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int **pages = list->pages;
  int nstencil = list->nstencil;
  int *stencil = list->stencil;
  int **stencilxyz = list->stencilxyz;

  int inum = 0;
  int npage = 0;
  int npnt = 0;

  // loop over owned & ghost atoms, storing neighbors

  for (i = 0; i < nall; i++) {

    if (pgsize - npnt < oneatom) {
      npnt = 0;
      npage++;
      if (npage == list->maxpage) pages = list->add_pages();
    }

    neighptr = &pages[npage][npnt];
    n = 0;

    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop over all atoms in surrounding bins in stencil including self
    // when i is a ghost atom, must check if stencil bin is out of bounds
    // skip i = j

    if (i < nlocal) {
      ibin = coord2bin(x[i]);
      for (k = 0; k < nstencil; k++) {
	for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
	  if (i == j) continue;
	  
	  jtype = type[j];
	  if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;
	
	  delx = xtmp - x[j][0];
	  dely = ytmp - x[j][1];
	  delz = ztmp - x[j][2];
	  rsq = delx*delx + dely*dely + delz*delz;
	
	  if (rsq <= cutneighsq[itype][jtype]) {
	    if (molecular) which = find_special(special[i],nspecial[i],tag[j]);
	    else which = 0;
	    if (which == 0) neighptr[n++] = j;
	    else if (which > 0) neighptr[n++] = which*nall + j;
	  }
	}
      }

    } else {
      ibin = coord2bin(x[i],xbin,ybin,zbin);
      for (k = 0; k < nstencil; k++) {
	xbin2 = xbin + stencilxyz[k][0];
	ybin2 = ybin + stencilxyz[k][1];
	zbin2 = zbin + stencilxyz[k][2];
	if (xbin2 < 0 || xbin2 >= mbinx ||
	    ybin2 < 0 || ybin2 >= mbiny ||
	    zbin2 < 0 || zbin2 >= mbinz) continue;
	for (j = binhead[ibin+stencil[k]]; j >= 0; j = bins[j]) {
	  if (i == j) continue;
	  
	  jtype = type[j];
	  if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;
	
	  delx = xtmp - x[j][0];
	  dely = ytmp - x[j][1];
	  delz = ztmp - x[j][2];
	  rsq = delx*delx + dely*dely + delz*delz;
	
	  if (rsq <= cutneighghostsq[itype][jtype]) {
	    if (molecular) which = find_special(special[i],nspecial[i],tag[j]);
	    else which = 0;
	    if (which == 0) neighptr[n++] = j;
	    else if (which > 0) neighptr[n++] = which*nall + j;
	  }
	}
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    npnt += n;
    if (n > oneatom || npnt >= pgsize)
      error->one("Neighbor list overflow, boost neigh_modify one or page");
  }

  list->inum = atom->nlocal;
  list->gnum = inum - atom->nlocal;
}

/* ----------------------------------------------------------------------
   binned neighbor list construction for all neighbors
   multi-type stencil is itype dependent and is distance checked
   every neighbor pair appears in list of both atoms i and j
------------------------------------------------------------------------- */

void Neighbor::full_multi(NeighList *list)
{
  int i,j,k,n,itype,jtype,ibin,which,ns;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *neighptr,*s;
  double *cutsq,*distsq;

  // bin local & ghost atoms

  bin_atoms();

  // loop over each atom, storing neighbors

  int **special = atom->special;
  int **nspecial = atom->nspecial;
  int *tag = atom->tag;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int *molecule = atom->molecule;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int molecular = atom->molecular;
  if (includegroup) nlocal = atom->nfirst;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int **pages = list->pages;
  int *nstencil_multi = list->nstencil_multi;
  int **stencil_multi = list->stencil_multi;
  double **distsq_multi = list->distsq_multi;

  int inum = 0;
  int npage = 0;
  int npnt = 0;

  for (i = 0; i < nlocal; i++) {

    if (pgsize - npnt < oneatom) {
      npnt = 0;
      npage++;
      if (npage == list->maxpage) pages = list->add_pages();
    }

    neighptr = &pages[npage][npnt];
    n = 0;

    itype = type[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop over all atoms in other bins in stencil, including self
    // skip if i,j neighbor cutoff is less than bin distance
    // skip i = j

    ibin = coord2bin(x[i]);
    s = stencil_multi[itype];
    distsq = distsq_multi[itype];
    cutsq = cutneighsq[itype];
    ns = nstencil_multi[itype];
    for (k = 0; k < ns; k++) {
      for (j = binhead[ibin+s[k]]; j >= 0; j = bins[j]) {
	jtype = type[j];
	if (cutsq[jtype] < distsq[k]) continue;
	if (i == j) continue;

	if (exclude && exclusion(i,j,itype,jtype,mask,molecule)) continue;

	delx = xtmp - x[j][0];
	dely = ytmp - x[j][1];
	delz = ztmp - x[j][2];
	rsq = delx*delx + dely*dely + delz*delz;

	if (rsq <= cutneighsq[itype][jtype]) {
	  if (molecular) which = find_special(special[i],nspecial[i],tag[j]);
	  else which = 0;
	  if (which == 0) neighptr[n++] = j;
	  else if (which > 0) neighptr[n++] = which*nall + j;
	}
      }
    }

    ilist[inum++] = i;
    firstneigh[i] = neighptr;
    numneigh[i] = n;
    npnt += n;
    if (n > oneatom || npnt >= pgsize)
      error->one("Neighbor list overflow, boost neigh_modify one or page");
  }

  list->inum = inum;
  list->gnum = 0;
}

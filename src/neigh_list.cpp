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
#include "lmptype.h"
#include "neigh_list.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

#define PGDELTA 1

enum{NSQ,BIN,MULTI};     // also in neighbor.cpp

/* ---------------------------------------------------------------------- */

NeighList::NeighList(LAMMPS *lmp, int size) : Pointers(lmp)
{
  maxatoms = 0;
  pgsize = size;

  inum = gnum = 0;
  ilist = NULL;
  numneigh = NULL;
  firstneigh = NULL;
  firstdouble = NULL;

  nndata_alloc_size = 0;
  nndataoffset = 0;
  nndata = 0;

  maxpage = 0;
  pages = NULL;
  dpages = NULL;
  dnum = 0;

  iskip = NULL;
  ijskip = NULL;

  listgranhistory = NULL;
  fix_history = NULL;

  respamiddle = 0;
  listinner = NULL;
  listmiddle = NULL;
  listfull = NULL;
  listcopy = NULL;
  listskip = NULL;

  maxstencil = 0;
  stencil = NULL;
  stencilxyz = NULL;

  maxstencil_multi = 0;
  nstencil_multi = NULL;
  stencil_multi = NULL;
  distsq_multi = NULL;
}

/* ---------------------------------------------------------------------- */

NeighList::~NeighList()
{
  if (!listcopy) {
    memory->destroy(ilist);
    memory->destroy(numneigh);
    memory->sfree(firstneigh);
    memory->sfree(firstdouble);

    for (int i = 0; i < maxpage; i++) memory->destroy(pages[i]);
    memory->sfree(pages);
    if (dnum) {
      for (int i = 0; i < maxpage; i++) memory->destroy(dpages[i]);
      memory->sfree(dpages);
    }
  }

  delete [] iskip;
  memory->destroy(ijskip);

  if (maxstencil) memory->destroy(stencil);
  if (ghostflag) memory->destroy(stencilxyz);

  if (maxstencil_multi) {
    for (int i = 1; i <= atom->ntypes; i++) {
      memory->destroy(stencil_multi[i]);
      memory->destroy(distsq_multi[i]);
    }
    delete [] nstencil_multi;
    delete [] stencil_multi;
    delete [] distsq_multi;
  }
}

/* ----------------------------------------------------------------------
   grow atom arrays to allow for nmax atoms
   triggered by more atoms on a processor
------------------------------------------------------------------------- */

void NeighList::grow(int nmax)
{
  // skip if grow not needed by this list
  // each list stores own maxatoms, b/c list->grow() called at different times
  // if list does not store neighbors of ghosts, compare nmax to maxatoms
  // else compare nlocal+nghost to maxatoms
  // if reset list size, set it to nmax

  if (!ghostflag && atom->nlocal <= maxatoms) return;
  if (ghostflag && atom->nlocal+atom->nghost <= maxatoms) return;
  maxatoms = nmax;

  memory->destroy(ilist);
  memory->destroy(numneigh);
  memory->destroy(firstneigh);
  memory->create(ilist,maxatoms,"neighlist:ilist");
  memory->create(numneigh,maxatoms,"neighlist:numneigh");
  firstneigh = (int **) memory->smalloc(maxatoms*sizeof(int *),
					"neighlist:firstneigh");

  clmattach(OCL_CONTEXT,ilist);
  clmattach(OCL_CONTEXT,numneigh);

  if (dnum) 
    firstdouble = (double **) memory->smalloc(maxatoms*sizeof(double *),
					      "neighlist:firstdouble");
}

/* ----------------------------------------------------------------------
   insure stencils are large enough for smax bins
   style = BIN or MULTI
------------------------------------------------------------------------- */

void NeighList::stencil_allocate(int smax, int style)
{
  int i;

  if (style == BIN) {
    if (smax > maxstencil) {
      maxstencil = smax;
      memory->destroy(stencil);
      memory->create(stencil,maxstencil,"neighlist:stencil");
      if (ghostflag) {
	memory->destroy(stencilxyz);
	memory->create(stencilxyz,maxstencil,3,"neighlist:stencilxyz");
      }
    }

  } else {
    int n = atom->ntypes;
    if (maxstencil_multi == 0) {
      nstencil_multi = new int[n+1];
      stencil_multi = new int*[n+1];
      distsq_multi = new double*[n+1];
      for (i = 1; i <= n; i++) {
	nstencil_multi[i] = 0;
	stencil_multi[i] = NULL;
	distsq_multi[i] = NULL;
      }
    }
    if (smax > maxstencil_multi) {
      maxstencil_multi = smax;
      for (i = 1; i <= n; i++) {
	memory->destroy(stencil_multi[i]);
	memory->destroy(distsq_multi[i]);
	memory->create(stencil_multi[i],maxstencil_multi,
		       "neighlist:stencil_multi");
	memory->create(distsq_multi[i],maxstencil_multi,
		       "neighlist:distsq_multi");
      }
    }
  }
}

/* ----------------------------------------------------------------------
   add PGDELTA pages to neighbor list
------------------------------------------------------------------------- */

int **NeighList::add_pages()
{
  int npage = maxpage;
  maxpage += PGDELTA;

  pages = (int **) 
    memory->srealloc(pages,maxpage*sizeof(int *),"neighlist:pages");
  for (int i = npage; i < maxpage; i++)
    memory->create(pages[i],pgsize,"neighlist:pages[i]");

  if (dnum) {
    dpages = (double **) 
      memory->srealloc(dpages,maxpage*sizeof(double *),"neighlist:dpages");
    for (int i = npage; i < maxpage; i++)
      memory->create(dpages[i],dnum*pgsize,"neighlist:dpages[i]");
  }

  return pages;
}

/* ----------------------------------------------------------------------
   copy skip info from request rq into list's iskip,ijskip
------------------------------------------------------------------------- */

void NeighList::copy_skip_info(int *rq_iskip, int **rq_ijskip)
{
  int ntypes = atom->ntypes;
  iskip = new int[ntypes+1];
  memory->create(ijskip,ntypes+1,ntypes+1,"neigh_list:ijskip");
  int i,j;
  for (i = 1; i <= ntypes; i++) iskip[i] = rq_iskip[i];
  for (i = 1; i <= ntypes; i++)
    for (j = 1; j <= ntypes; j++)
      ijskip[i][j] = rq_ijskip[i][j];
}

/* ----------------------------------------------------------------------
   print attributes of this list and associated request
------------------------------------------------------------------------- */

void NeighList::print_attributes()
{
  if (comm->me != 0) return;

  NeighRequest *rq = neighbor->requests[index];

  printf("Neighbor list/request %d:\n",index);
  printf("  %d = build flag\n",buildflag);
  printf("  %d = grow flag\n",growflag);
  printf("  %d = stencil flag\n",stencilflag);
  printf("  %d = ghost flag\n",ghostflag);
  printf("\n");
  printf("  %d = pair\n",rq->pair);
  printf("  %d = fix\n",rq->fix);
  printf("  %d = compute\n",rq->compute);
  printf("  %d = command\n",rq->command);
  printf("\n");
  printf("  %d = half\n",rq->half);
  printf("  %d = full\n",rq->full);
  printf("  %d = gran\n",rq->gran);
  printf("  %d = granhistory\n",rq->granhistory);
  printf("  %d = respainner\n",rq->respainner);
  printf("  %d = respamiddle\n",rq->respamiddle);
  printf("  %d = respaouter\n",rq->respaouter);
  printf("  %d = half_from_full\n",rq->half_from_full);
  printf("\n");
  printf("  %d = occasional\n",rq->occasional);
  printf("  %d = dnum\n",rq->dnum);
  printf("  %d = ghost\n",rq->ghost);
  printf("  %d = copy\n",rq->copy);
  printf("  %d = skip\n",rq->skip);
  printf("  %d = otherlist\n",rq->otherlist);
  printf("  %p = listskip\n",listskip);
  printf("\n");
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
   if growflag = 0, maxatoms & maxpage will also be 0
   if stencilflag = 0, maxstencil * maxstencil_multi will also be 0
------------------------------------------------------------------------- */

bigint NeighList::memory_usage()
{
  bigint bytes = 0;
  bytes += memory->usage(ilist,maxatoms);
  bytes += memory->usage(numneigh,maxatoms);
  bytes += maxatoms * sizeof(int *);
  bytes += memory->usage(pages,maxpage,pgsize);

  if (dnum) {
    bytes += maxatoms * sizeof(double *);
    bytes += memory->usage(dpages,maxpage,dnum*pgsize);
  }

  if (maxstencil) bytes += memory->usage(stencil,maxstencil);
  if (ghostflag) bytes += memory->usage(stencilxyz,maxstencil,3);

  if (maxstencil_multi) {
    bytes += memory->usage(stencil_multi,atom->ntypes,maxstencil_multi);
    bytes += memory->usage(distsq_multi,atom->ntypes,maxstencil_multi);
  }

  return bytes;
}

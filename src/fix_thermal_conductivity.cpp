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

#include "math.h"
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "fix_thermal_conductivity.h"
#include "atom.h"
#include "force.h"
#include "domain.h"
#include "error.h"

using namespace LAMMPS_NS;

#define BIG 1.0e10

#define MIN(A,B) ((A) < (B)) ? (A) : (B)
#define MAX(A,B) ((A) > (B)) ? (A) : (B)

/* ---------------------------------------------------------------------- */

FixThermalConductivity::FixThermalConductivity(LAMMPS *lmp,
					       int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 6) error->all("Illegal fix thermal/conductivity command");

  MPI_Comm_rank(world,&me);

  nevery = atoi(arg[3]);
  if (nevery <= 0) error->all("Illegal fix thermal/conductivity command");

  scalar_flag = 1;
  scalar_vector_freq = nevery;
  extscalar = 0;

  if (strcmp(arg[4],"x") == 0) edim = 0;
  else if (strcmp(arg[4],"y") == 0) edim = 1;
  else if (strcmp(arg[4],"z") == 0) edim = 2;
  else error->all("Illegal fix thermal/conductivity command");

  nbin = atoi(arg[5]);
  if (nbin < 3) error->all("Illegal fix thermal/conductivity command");

  // optional keywords

  nswap = 1;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"swap") == 0) {
      if (iarg+2 > narg)
	error->all("Illegal fix thermal/conductivity command");
      nswap = atoi(arg[iarg+1]);
      if (nswap <= 0)
	error->all("Fix thermal/conductivity swap value must be positive");
      iarg += 2;
    } else error->all("Illegal fix thermal/conductivity command");
  }

  // initialize array sizes to nswap+1 so have space to shift values down

  index_lo = new int[nswap+1];
  index_hi = new int[nswap+1];
  ke_lo = new double[nswap+1];
  ke_hi = new double[nswap+1];

  e_exchange = 0.0;
}

/* ---------------------------------------------------------------------- */

FixThermalConductivity::~FixThermalConductivity()
{
  delete [] index_lo;
  delete [] index_hi;
  delete [] ke_lo;
  delete [] ke_hi;
}

/* ---------------------------------------------------------------------- */

int FixThermalConductivity::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixThermalConductivity::init()
{
  // set bounds of 2 slabs in edim
  // only necessary for static box, else re-computed in end_of_step()
  // lo bin is always bottom bin
  // if nbin even, hi bin is just below half height
  // if nbin odd, hi bin straddles half height

  if (domain->box_change == 0) {
    prd = domain->prd[edim];
    boxlo = domain->boxlo[edim];
    boxhi = domain->boxhi[edim];
    double binsize = (boxhi-boxlo) / nbin;
    slablo_lo = boxlo;
    slablo_hi = boxlo + binsize;
    slabhi_lo = boxlo + ((nbin-1)/2)*binsize;
    slabhi_hi = boxlo + ((nbin-1)/2 + 1)*binsize;
  }

  periodicity = domain->periodicity[edim];
}

/* ---------------------------------------------------------------------- */

void FixThermalConductivity::end_of_step()
{
  int i,j,m,insert;
  double p,coord,ke;
  MPI_Status status;
  struct {
    double value;
    int proc;
  } mine[2],all[2];

  // if box changes, recompute bounds of 2 slabs in edim

  if (domain->box_change) {
    prd = domain->prd[edim];
    boxlo = domain->boxlo[edim];
    boxhi = domain->boxhi[edim];
    double binsize = (boxhi-boxlo) / nbin;
    slablo_lo = boxlo;
    slablo_hi = boxlo + binsize;
    slabhi_lo = boxlo + ((nbin-1)/2)*binsize;
    slabhi_hi = boxlo + ((nbin-1)/2 + 1)*binsize;
  }

  // make 2 lists of up to nswap atoms
  // hottest atoms in lo slab, coldest atoms in hi slab (really mid slab)
  // lo slab list is sorted by hottest, hi slab is sorted by coldest
  // map atoms back into periodic box if necessary
  // insert = location in list to insert new atom

  double **x = atom->x;
  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  nhi = nlo = 0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      coord = x[i][edim];
      if (coord < boxlo && periodicity) coord += prd;
      else if (coord >= boxhi && periodicity) coord -= prd;

      if (coord >= slablo_lo && coord < slablo_hi) {
	ke = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
	if (mass) ke *= 0.5*mass[type[i]];
	else ke *= 0.5*rmass[i];
	if (nlo < nswap || ke > ke_lo[nswap-1]) {
	  for (insert = nlo-1; insert >= 0; insert--)
	    if (ke < ke_lo[insert]) break;
	  insert++;
	  for (m = nlo-1; m >= insert; m--) {
	    ke_lo[m+1] = ke_lo[m];
	    index_lo[m+1] = index_lo[m];
	  }
	  ke_lo[insert] = ke;
	  index_lo[insert] = i;
	  if (nlo < nswap) nlo++;
	}
      }

      if (coord >= slabhi_lo && coord < slabhi_hi) {
	ke = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];
	if (mass) ke *= 0.5*mass[type[i]];
	else ke *= 0.5*rmass[i];
	if (nhi < nswap || ke < ke_hi[nswap-1]) {
	  for (insert = nhi-1; insert >= 0; insert--)
	    if (ke > ke_hi[insert]) break;
	  insert++;
	  for (m = nhi-1; m >= insert; m--) {
	    ke_hi[m+1] = ke_hi[m];
	    index_hi[m+1] = index_hi[m];
	  }
	  ke_hi[insert] = ke;
	  index_hi[insert] = i;
	  if (nhi < nswap) nhi++;
	}
      }
    }

  // loop over nswap pairs
  // pair 2 global atoms at beginning of sorted lo/hi slab lists via Allreduce
  // BIG values are for procs with no atom to contribute
  // use negative of hottest KE since is doing a MINLOC
  // MINLOC also communicates which procs own them
  // exchange kinetic energy between the 2 particles
  // if I own both particles just swap, else point2point comm of velocities

  double sbuf[3],rbuf[3];

  mine[0].proc = mine[1].proc = me;
  int ilo = 0;
  int ihi = 0;

  for (m = 0; m < nswap; m++) {
    if (ilo < nlo) mine[0].value = -ke_lo[ilo];
    else mine[0].value = BIG;
    if (ihi < nhi) mine[1].value = ke_hi[ihi];
    else mine[1].value = BIG;
    
    MPI_Allreduce(mine,all,2,MPI_DOUBLE_INT,MPI_MINLOC,world);
    if (all[0].value == BIG || all[1].value == BIG) continue;
    all[0].value = -all[0].value;
    e_exchange += force->mvv2e * (all[0].value - all[1].value);

    if (me == all[0].proc && me == all[1].proc) {
      i = index_lo[ilo++];
      j = index_hi[ihi++];
      rbuf[0] = v[i][0];
      rbuf[1] = v[i][1];
      rbuf[2] = v[i][2];
      v[i][0] = v[j][0];
      v[i][1] = v[j][1];
      v[i][2] = v[j][2];
      v[j][0] = rbuf[0];
      v[j][1] = rbuf[1];
      v[j][2] = rbuf[2];
      
    } else if (me == all[0].proc) {
      i = index_lo[ilo++];
      sbuf[0] = v[i][0];
      sbuf[1] = v[i][1];
      sbuf[2] = v[i][2];
      MPI_Sendrecv(sbuf,3,MPI_DOUBLE,all[1].proc,0,
		   rbuf,3,MPI_DOUBLE,all[1].proc,0,world,&status);
      v[i][0] = rbuf[0];
      v[i][1] = rbuf[1];
      v[i][2] = rbuf[2];

    } else if (me == all[1].proc) {
      j = index_hi[ihi++];
      sbuf[0] = v[j][0];
      sbuf[1] = v[j][1];
      sbuf[2] = v[j][2];
      MPI_Sendrecv(sbuf,3,MPI_DOUBLE,all[0].proc,0,
		   rbuf,3,MPI_DOUBLE,all[0].proc,0,world,&status);
      v[j][0] = rbuf[0];
      v[j][1] = rbuf[1];
      v[j][2] = rbuf[2];
    }
  }
}

/* ---------------------------------------------------------------------- */

double FixThermalConductivity::compute_scalar()
{
  return e_exchange;
}
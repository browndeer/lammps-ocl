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


#ifndef LMP_NEIGH_REQUEST_H
#define LMP_NEIGH_REQUEST_H

#include "pointers.h"

namespace LAMMPS_NS {

class NeighRequest : protected Pointers {
 public:
  void *requestor;       // class that made request
  int id;                // ID of request
                         // used to track multiple requests from one class

  // which class is requesting the list, one flag is 1, others are 0

  int pair;
  int fix;
  int compute;
  int command;

  // kind of list requested
  // set by requesting class

  int half;              // 1 if half neigh list
  int full;              // 1 if full neigh list

  int use_ocl;

  int gran;              // 1 if granular list
  int granhistory;       // 1 if granular history list

  int respainner;        // 1 if a rRESPA inner list        
  int respamiddle;       // 1 if a rRESPA middle list
  int respaouter;        // 1 if a rRESPA outer list

  int half_from_full;    // 1 if half list computed from previous full list

  // 0 if needed every reneighboring during run
  // 1 if occasionally needed by a fix, compute, etc
  // set by requesting class

  int occasional;

  // 0 if use force::newton_pair setting
  // 1 if override with pair newton on
  // 2 if override with pair newton off

  int newton;

  // number of auxiliary floating point values to store, 0 if none
  // set by requesting class

  int dnum;

  // 1 if also need neighbors of ghosts

  int ghost;

  // set by neighbor and pair_hybrid after all requests are made
  // these settings do not change kind value

  int copy;              // 1 if this list copied from another list

  int skip;              // 1 if this list skips atom types from another list
  int *iskip;            // iskip[i] if atoms of type I are not in list
  int **ijskip;          // ijskip[i][j] if pairs of type I,J are not in list

  int otherlist;         // other list to copy or skip from

  // methods

  NeighRequest(class LAMMPS *);
  ~NeighRequest();
  int identical(NeighRequest *);
  int same_kind(NeighRequest *);
  int same_skip(NeighRequest *);
  void copy_request(NeighRequest *);
};

}

#endif

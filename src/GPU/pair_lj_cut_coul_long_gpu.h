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

#ifdef PAIR_CLASS

PairStyle(lj/cut/coul/long/gpu,PairLJCutCoulLongGPU)

#else

#ifndef LMP_PAIR_LJ_CUT_COUL_LONG_GPU_H
#define LMP_PAIR_LJ_CUT_COUL_LONG_GPU_H

#include "pair_lj_cut_coul_long.h"

namespace LAMMPS_NS {

class PairLJCutCoulLongGPU : public PairLJCutCoulLong {
 public:
  PairLJCutCoulLongGPU(LAMMPS *lmp);
  ~PairLJCutCoulLongGPU();
  void cpu_compute(int, int, int);
  void cpu_compute(int *, int, int, int);
  void compute(int, int);
  void init_style();
  double memory_usage();

 enum { GPU_PAIR, GPU_NEIGH };

 private:
  int gpu_mode;
  double cpu_time;
  int *gpulist;
};

}
#endif
#endif


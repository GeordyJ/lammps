/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(wall/region/tjatjopoulos,FixWallRegionTjat);
// clang-format on
#else

#ifndef LMP_FIX_WALL_REGION_TJAT_H
#define LMP_FIX_WALL_REGION_TJAT_H

#include "fix.h"

namespace LAMMPS_NS {

class FixWallRegionTjat : public Fix {
 public:
  FixWallRegionTjat(class LAMMPS *, int, char **);
  ~FixWallRegionTjat() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  double compute_scalar() override;
  double compute_vector(int) override;

 private:
  int style;
  double epsilon, sigma, cutoff, rho_A; //Surface density
  double R, R2; //radius and its square of cylindrical region
  double tjat_coeff, psi6_coeff, psi3_coeff;
  double psi6_der1, psi6_der2, psi3_der1, psi3_der2;

  int eflag;
  double ewall[4], ewall_all[4];
  int ilevel_respa;
  char *idregion;
  class Region *region;

  double eng, fwall;

  void tjatjopoulos(double);
};

}    // namespace LAMMPS_NS

#endif
#endif

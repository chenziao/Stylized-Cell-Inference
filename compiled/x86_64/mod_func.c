#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _AlphaSynapse1_reg(void);
extern void _NaTa_t_reg(void);
extern void _SKv3_1_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," \"mechanisms//AlphaSynapse1.mod\"");
    fprintf(stderr," \"mechanisms//NaTa_t.mod\"");
    fprintf(stderr," \"mechanisms//SKv3_1.mod\"");
    fprintf(stderr, "\n");
  }
  _AlphaSynapse1_reg();
  _NaTa_t_reg();
  _SKv3_1_reg();
}

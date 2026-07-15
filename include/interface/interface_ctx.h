#ifndef __LagRBF_interface_ctx_h
#define __LagRBF_interface_ctx_h

#include "lagrangian_rbf.h"

LAGRBF_EXPORT void *lagrbf_create();
LAGRBF_EXPORT void lagrbf_destroy(void *ctx);

#endif
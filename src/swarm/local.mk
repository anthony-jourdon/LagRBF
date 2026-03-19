liblagrangian-offline-y.c += $(call thisdir, \
  swarm_utils.c \
  swarm_interp.c \
  swarm_output.c \
  swarm_data.c \
  swarm_load.c \
)

ifeq ($(HAVE_AVX512),1)
liblagrangian-offline-y.c += $(call thisdir, \
  weights_avx512.c \
  weights_avx.c \
)
else ifeq ($(HAVE_AVX256),1)
liblagrangian-offline-y.c += $(call thisdir, \
  weights_avx.c \
)
endif
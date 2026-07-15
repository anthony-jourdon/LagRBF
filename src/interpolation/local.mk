liblagrbf-offline-y.c += $(call thisdir, \
	interpolation.c \
)

ifeq ($(HAVE_AVX512),1)
liblagrbf-offline-y.c += $(call thisdir, \
  interpolation_avx.c \
	interpolation_avx512.c \
)
else ifeq ($(HAVE_AVX256),1)
liblagrbf-offline-y.c += $(call thisdir, \
  interpolation_avx.c \
)
endif
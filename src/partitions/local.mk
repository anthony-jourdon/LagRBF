liblagrbf-offline-y.c += $(call thisdir, \
	partition.c \
	partition_solve.c \
	partition_factor.c \
	partition_ksp.c \
	partition_io.c \
)

ifeq ($(HAVE_AVX512),1)
liblagrbf-offline-y.c += $(call thisdir, \
  partition_asmb_avx.c \
  partition_asmb_avx512.c \
)
else ifeq ($(HAVE_AVX256),1)
liblagrbf-offline-y.c += $(call thisdir, \
  partition_asmb_avx.c \
)
endif
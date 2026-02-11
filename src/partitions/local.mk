liblagrangian-offline-y.c += $(call thisdir, \
	partition.c \
	partition_solve.c \
	partition_factor.c \
	partition_ksp.c \
	partition_io.c \
	partition_asmb_avx.c \
)
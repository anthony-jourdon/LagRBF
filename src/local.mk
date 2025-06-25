liblagrangian-offline-y.c += $(call thisdir, \
	lagrangian_rbf_init.c \
	lagrangian_rbf.c \
	partition_of_unity.c \
	partition.c \
	partition_solve.c \
	kernels_rbf.c \
	io_utils.c \
	geodynamics_fields.c \
	swarm_utils.c \
	swarm_interp.c \
	swarm_data.c \
	interpolation.c \
	interpolation_avx.c \
)

lagrangian-offline-drivers-y.c += $(call thisdir, \
)

# include subdirectories
include $(call incsubdirs, tests)
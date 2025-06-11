liblagrangian-offline-y.c += $(call thisdir, \
	partition_of_unity.c \
	partition.c \
	partition_solve.c \
	kernels_rbf.c \
	io_utils.c \
)

lagrangian-offline-drivers-y.c += $(call thisdir, \
)

# include subdirectories
include $(call incsubdirs, tests)
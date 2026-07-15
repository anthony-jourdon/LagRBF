liblagrbf-offline-y.c += $(call thisdir, \
	reader_utils.c \
)

# include subdirectories
include $(call incsubdirs, stagYY petsc_loader)
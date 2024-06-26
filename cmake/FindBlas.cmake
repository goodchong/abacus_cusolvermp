if(DEFINED BLAS_DIR)
    string(APPEND CMAKE_PREFIX_PATH ";${BLAS_DIR}")
endif()
if(DEFINED BLAS_LIBRARY)
    set(BLAS_LIBRARIES ${BLAS_LIBRARY})
endif()

find_package(BLAS REQUIRED)

if(NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS UNKNOWN IMPORTED)
    set_target_properties(BLAS::BLAS PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
	IMPORTED_LOCATION "${BLAS_LIBRARIES}")
endif()

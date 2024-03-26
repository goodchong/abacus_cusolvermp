# In compatibility to builtin FindLAPACK.cmake before v3.5.4
if(DEFINED LAPACK_DIR)
  string(APPEND CMAKE_PREFIX_PATH ";${LAPACK_DIR}")
endif()
if(DEFINED LAPACK_LIBRARY)
  set(LAPACK_LIBRARIES ${LAPACK_LIBRARY})
endif()

find_package(Blas REQUIRED)
find_package(LAPACK REQUIRED)

if(NOT TARGET LAPACK::LAPACK)
    add_library(LAPACK::LAPACK UNKNOWN IMPORTED)
    set_target_properties(LAPACK::LAPACK PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${LAPACK_LIBRARIES}")
endif()
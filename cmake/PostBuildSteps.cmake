# Post-build steps previously handled by setup.py's build_ext.run().
# These run as CMake install(SCRIPT) or install(CODE) commands.

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()
if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include)
endif()

# --- Header wrapping with TORCH_STABLE_ONLY guards ---
# Wrap installed headers so they error when included with TORCH_STABLE_ONLY
# or TORCH_TARGET_VERSION defined. This is done at install time via a script.
install(CODE "
  execute_process(
    COMMAND \"${Python_EXECUTABLE}\"
      \"${PROJECT_SOURCE_DIR}/tools/wrap_headers.py\"
      \"\${CMAKE_INSTALL_PREFIX}/${TORCH_INSTALL_INCLUDE_DIR}\"
  )
")

# --- Compile commands merging ---
# Merge compile_commands.json from build subdirectories.
add_custom_target(merge_compile_commands ALL
  COMMAND "${Python_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/tools/merge_compile_commands.py"
    "${CMAKE_BINARY_DIR}" "${PROJECT_SOURCE_DIR}"
  COMMENT "Merging compile_commands.json..."
  VERBATIM
)

import shutil

# Copy drlDLL.lib
shutil.copy2('./pre_compiled_libs/drlDLL.lib', '../or-tools-6.7.2-customized/ortools/constraint_solver')

# Copy the dlls
shutil.copy2('./pre_compiled_libs/drlDLL.dll', '../or-tools-6.7.2-customized/ortools/gen/ortools/constraint_solver')
shutil.copy2('./pre_compiled_libs/cublas64_100.dll', '../or-tools-6.7.2-customized/ortools/gen/ortools/constraint_solver')
shutil.copy2('./pre_compiled_libs/cudart64_100.dll', '../or-tools-6.7.2-customized/ortools/gen/ortools/constraint_solver')
shutil.copy2('./pre_compiled_libs/cusparse64_100.dll', '../or-tools-6.7.2-customized/ortools/gen/ortools/constraint_solver')
shutil.copy2('./pre_compiled_libs/libiomp5md.dll', '../or-tools-6.7.2-customized/ortools/gen/ortools/constraint_solver')
shutil.copy2('./pre_compiled_libs/tbb.dll', '../or-tools-6.7.2-customized/ortools/gen/ortools/constraint_solver')

print('Pre-compiled files copied.')
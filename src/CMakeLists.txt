# Update to CMakeLists.txt - Add new files to llama library
llama_add_compile_flags()

#
# libraries
#

# llama

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-memory.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampling.cpp
            llama-vocab.cpp
            graph_reasoning.cpp
            finetune.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            )

target_include_directories(llama PUBLIC . ../include)
target_compile_features   (llama PUBLIC cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()

# Add to the main executable target (assuming it's named 'main')
# You'll need to modify this section based on how your main target is defined
if(BUILD_MAIN)
    add_executable(main main.cpp main_finetune.cpp)
    target_link_libraries(main PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
endif()

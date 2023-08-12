cmake_minimum_required(VERSION 3.20)

project(spam CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glslang REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main
	glfw
	${Vulkan_LIBRARIES}
	SPIRV
	# glslang::glslang
	glslang::glslang-default-resource-limits
	OpenMP::OpenMP_CXX
)

include_directories(.
	glm
	thirdparty/
	thirdparty/littlevk/
	${OpenMP_CXX_INCLUDE_DIRS}
)

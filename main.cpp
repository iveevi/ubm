#include <iostream>

#include <omp.h>

#include <littlevk/littlevk.hpp>

static constexpr size_t resolution = 256;

// Shader sources
const std::string vertex_shader_source = R"(
#version 450

layout (location = 0) in vec2 position;

void main()
{
	gl_Position = vec4(position, 0.0, 1.0);
}
)";

// Use cppsl to parametrize the fragment shader by resolution
struct push_constants {
	float center[2];
	float extent[2];
	// glm::vec2 resolution;
};

const std::string fragment_shader_source = R"(
#version 450

layout (location = 0) out vec4 fragment;

// Density array
layout (binding = 0) uniform sampler2D density_sampler;

// Push constants to control viewport
layout (push_constant) uniform PushConstants {
	vec2 center;
	vec2 extent;
	// vec2 resolution;
};

// Viridis colormap (assume normalized [0, 1] inputs)
vec3 viridis(float x)
{
	vec3 palette[5] = vec3[5] (
		vec3(68, 1, 84) / 255.0,
		vec3(59, 82, 139) / 255.0,
		vec3(33, 145, 140) / 255.0,
		vec3(94, 201, 98) / 255.0,
		vec3(253, 231, 37) / 255.0
	);

	// Find the interval, then interpolate
	x = clamp(x, 0, 1);
	float x_fl = floor(x * 4);
	float x_cl = ceil(x * 4);
	float t = (x * 4) - x_fl;
	t = t * t * (3 - 2 * t);
	return mix(palette[int(x_fl)], palette[int(x_cl)], t);
}

void main() {
	// vec2 center = vec2(0, 0);
	// vec2 extent = vec2(1, 1);
	vec2 resolution = vec2(256, 256);
	vec2 uv = gl_FragCoord.xy / vec2(1024, 1024);

	vec2 xy = (uv - center) / extent;
	if (xy.x > 1 || xy.x < 0 || xy.y > 1 || xy.y < 0)
		discard;

	// Distance to closest grid line
	vec2 grid_xy = xy * resolution;

	float x_grid_fl = floor(grid_xy.x);
	float x_grid_cl = ceil(grid_xy.x);
	float x_dist = min(abs(grid_xy.x - x_grid_fl), abs(grid_xy.x - x_grid_cl));

	float y_grid_fl = floor(grid_xy.y);
	float y_grid_cl = ceil(grid_xy.y);
	float y_dist = min(abs(grid_xy.y - y_grid_fl), abs(grid_xy.y - y_grid_cl));

	float gld = min(x_dist, y_dist);
	gld = smoothstep(0, 1, pow(1 - gld, 16));

	// If the width is less than a pixel, ignore
	float pot = pow(0.25, 1/8);
	pot *= 2 * extent.x;
	if (pot < 5.0)
		gld *= pow(pot/5.0, 8);

	float density = 0;
	int ix = int(x_grid_fl);
	int iy = int(y_grid_fl);
	if (ix >= 0 && ix < 256 && iy >= 0 && iy < 256)
		density = texelFetch(density_sampler, ivec2(ix, iy), 0).r;

	vec3 heat = viridis(density);

	fragment = vec4(vec3(gld) + heat, 1.0);
}
)";

// Diffusion backend
struct Array {
	float density[resolution * resolution];
	float dual[resolution * resolution];

	void randomize() {
		for (size_t i = 0; i < resolution * resolution; i++) {
			int r = rand() % 100;
			density[i] = (float) r / 100.0f;
		}
	}

	void diffuse() {
		// Diffuse the density
		#pragma omp parallel for
		for (size_t i = 0; i < resolution * resolution; i++) {
			float d = density[i];
			float d_l = i % resolution == 0 ? 0 : density[i - 1];
			float d_r = i % resolution == resolution - 1 ? 0 : density[i + 1];
			float d_u = i < resolution ? 0 : density[i - resolution];
			float d_d = i >= resolution * (resolution - 1) ? 0 : density[i + resolution];

			dual[i] = d + 0.25f * (d_l + d_r + d_u + d_d - 4 * d);
		}

		std::swap(density, dual);
	}

	void add(float x, float y, float amount, float radius) {
		// Add density at (x, y) with radius
		float x0 = x - radius;
		float x1 = x + radius;
		float y0 = y - radius;
		float y1 = y + radius;

		int ix0 = std::max(0, (int) (x0 * resolution));
		int ix1 = std::min((int) (x1 * resolution), (int) resolution - 1);
		int iy0 = std::max(0, (int) (y0 * resolution));
		int iy1 = std::min((int) (y1 * resolution), (int) resolution - 1);

		for (int ix = ix0; ix <= ix1; ix++) {
			for (int iy = iy0; iy <= iy1; iy++) {
				float dx = (float) ix / resolution - x;
				float dy = (float) iy / resolution - y;
				float d = sqrt(dx * dx + dy * dy);
				if (d < radius) {
					float &dref = density[ix + iy * resolution];
					dref += amount * (1 - d / radius);
					dref = std::max(dref, 0.0f);
				}
			}
		}
	}
} g_array;

// Rendering backend
struct Rendering : littlevk::Skeleton {
	static constexpr size_t WIDTH = 1024;
	static constexpr size_t HEIGHT = 1024;

	// Vulkan resources
	vk::PhysicalDevice phdev;
	vk::PhysicalDeviceMemoryProperties mem_props;

	littlevk::Deallocator *dal;
	littlevk::Buffer grid_vertices;
	littlevk::Buffer grid_indices;

	vk::Sampler sampler;
	littlevk::Image density_image;
	littlevk::Buffer density_staging;

	vk::RenderPass render_pass;
	vk::Pipeline pipeline;
	vk::PipelineLayout pipeline_layout;

	vk::DescriptorSetLayout descriptor_set_layout;
	vk::DescriptorPool descriptor_pool;
	vk::DescriptorSet descriptor_set;

	std::vector <vk::Framebuffer> framebuffers;

	vk::CommandPool command_pool;
	std::vector <vk::CommandBuffer> command_buffers;

	littlevk::PresentSyncronization sync;

	// Viewport state
	push_constants constants;

	// Vertex properties
	static constexpr vk::VertexInputBindingDescription vertex_binding {
		0, 2 * sizeof(float), vk::VertexInputRate::eVertex
	};

	static constexpr std::array <vk::VertexInputAttributeDescription, 1> vertex_attributes {
		vk::VertexInputAttributeDescription {
			0, 0, vk::Format::eR32G32Sfloat, 0
		},
	};

	// Derfault sampler info
	static constexpr vk::SamplerCreateInfo default_sampler_info {
		vk::SamplerCreateFlags {},
		vk::Filter::eLinear,
		vk::Filter::eLinear,
		vk::SamplerMipmapMode::eLinear,
		vk::SamplerAddressMode::eRepeat,
		vk::SamplerAddressMode::eRepeat,
		vk::SamplerAddressMode::eRepeat,
		0.0f,
		VK_FALSE,
		1.0f,
		VK_FALSE,
		vk::CompareOp::eAlways,
		0.0f,
		0.0f,
		vk::BorderColor::eIntOpaqueBlack,
		VK_FALSE
	};

	// Initialize the rendering backend from a physical device
	void from(const vk::PhysicalDevice &phdev) {
		// Copy physical device
		this->phdev = phdev;
		mem_props = phdev.getMemoryProperties();

		// Initialize skeleton
		skeletonize(phdev, { WIDTH, HEIGHT }, "Diffusion");

		// Create the deallocator
		dal = new littlevk::Deallocator(device);

		// Create the render pass
		std::array <vk::AttachmentDescription, 1> attachments {
			vk::AttachmentDescription {
				{},
				swapchain.format,
				vk::SampleCountFlagBits::e1,
				vk::AttachmentLoadOp::eClear,
				vk::AttachmentStoreOp::eStore,
				vk::AttachmentLoadOp::eDontCare,
				vk::AttachmentStoreOp::eDontCare,
				vk::ImageLayout::eUndefined,
				vk::ImageLayout::ePresentSrcKHR,
			},
		};

		std::array <vk::AttachmentReference, 1> color_attachments {
			vk::AttachmentReference {
				0, vk::ImageLayout::eColorAttachmentOptimal,
			}
		};

		vk::SubpassDescription subpass {
			{}, vk::PipelineBindPoint::eGraphics,
			{}, color_attachments,
		};

		render_pass = littlevk::render_pass(
			device,
			vk::RenderPassCreateInfo {
				{}, attachments, subpass
			}
		).unwrap(dal);

		// Create framebuffers from the swapchain
		littlevk::FramebufferSetInfo fb_info;
		fb_info.swapchain = &swapchain;
		fb_info.render_pass = render_pass;
		fb_info.extent = window->extent;

		framebuffers = littlevk::framebuffers(device, fb_info).unwrap(dal);

		// Allocate command buffers
		command_pool = littlevk::command_pool(device,
			vk::CommandPoolCreateInfo {
				vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				littlevk::find_graphics_queue_family(phdev)
			}
		).unwrap(dal);

		command_buffers = device.allocateCommandBuffers({
			command_pool, vk::CommandBufferLevel::ePrimary, 2
		});

		// Screen quad vertices
		std::array <float, 8> vertices {
			-1.0f, -1.0f,
			 1.0f, -1.0f,
			 1.0f,  1.0f,
			-1.0f,  1.0f,
		};

		std::array <uint32_t, 6> indices {
			0, 1, 2, 2, 3, 0
		};

		// grid_vertices = littlevk::buffer(device,
		// 	vertices.size() * sizeof(float),
		// 	vk::BufferUsageFlagBits::eVertexBuffer,
		// 	mem_props
		// ).unwrap(dal);
		//
		// grid_indices = littlevk::buffer(device,
		// 	indices.size() * sizeof(uint32_t),
		// 	vk::BufferUsageFlagBits::eIndexBuffer,
		// 	mem_props
		// ).unwrap(dal);
		//
		// littlevk::upload(device, grid_vertices, vertices);
		// littlevk::upload(device, grid_indices, indices);

		grid_vertices = littlevk::buffer(device, vertices, vk::BufferUsageFlagBits::eVertexBuffer, mem_props).unwrap(dal);
		grid_indices = littlevk::buffer(device, indices, vk::BufferUsageFlagBits::eIndexBuffer, mem_props).unwrap(dal);

		// Compile shader modules
		vk::ShaderModule vertex_module = littlevk::shader::compile(
			device, vertex_shader_source,
			vk::ShaderStageFlagBits::eVertex
		).unwrap(dal);

		vk::ShaderModule fragment_module = littlevk::shader::compile(
			device, fragment_shader_source,
			vk::ShaderStageFlagBits::eFragment
		).unwrap(dal);

		// Allocate a descriptor pool with one descriptor set
		vk::DescriptorPoolSize pool_size {
			vk::DescriptorType::eCombinedImageSampler, 1
		};

		descriptor_pool = littlevk::descriptor_pool(
			device, vk::DescriptorPoolCreateInfo {
				{}, 1, pool_size
			}
		).unwrap(dal);

		// Create the pipeline
		vk::DescriptorSetLayoutBinding dsl_binding {
			0, vk::DescriptorType::eCombinedImageSampler,
			1, vk::ShaderStageFlagBits::eFragment
		};

		descriptor_set_layout = littlevk::descriptor_set_layout(
			device, vk::DescriptorSetLayoutCreateInfo {
				{}, dsl_binding
			}
		).unwrap(dal);

		vk::PushConstantRange push_constant_range {
			vk::ShaderStageFlagBits::eFragment,
			0, sizeof(push_constants)
		};

		constants.center[0] = 0;
		constants.center[1] = 0;
		constants.extent[0] = 1;
		constants.extent[1] = 1;

		pipeline_layout = littlevk::pipeline_layout(
			device,
			vk::PipelineLayoutCreateInfo {
				{}, descriptor_set_layout, push_constant_range
			}
		).unwrap(dal);

		littlevk::pipeline::GraphicsCreateInfo pipeline_info;
		pipeline_info.vertex_binding = vertex_binding;
		pipeline_info.vertex_attributes = vertex_attributes;
		pipeline_info.vertex_shader = vertex_module;
		pipeline_info.fragment_shader = fragment_module;
		pipeline_info.extent = window->extent;
		pipeline_info.pipeline_layout = pipeline_layout;
		pipeline_info.render_pass = render_pass;

		pipeline = littlevk::pipeline::compile(device, pipeline_info).unwrap(dal);

		// Create the syncronization objects
		sync = littlevk::present_syncronization(device, 2).unwrap(dal);

		// Prepare density rendering resources
		sampler = littlevk::sampler(device, default_sampler_info).unwrap(dal);

		density_staging = littlevk::buffer(device,
			resolution * resolution * sizeof(float),
			vk::BufferUsageFlagBits::eTransferSrc,
			mem_props
		).unwrap(dal);

		// littlevk::upload(device, density_staging, g_array.density);

		density_image = littlevk::image(device, {
			(uint32_t) resolution,
			(uint32_t) resolution,
			vk::Format::eR32Sfloat,
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
			vk::ImageAspectFlagBits::eColor
		}, mem_props).unwrap(dal);

		// littlevk::submit_now(device, command_pool, graphics_queue,
		// 	[&](const vk::CommandBuffer &cmd) {
		// 		littlevk::transition(cmd, density_image,
		// 			vk::ImageLayout::eUndefined,
		// 			vk::ImageLayout::eTransferDstOptimal);
		//
		// 		littlevk::copy_buffer_to_image(cmd,
		// 			density_image, density_staging,
		// 			vk::ImageLayout::eTransferDstOptimal);
		//
		// 		littlevk::transition(cmd,
		// 			density_image,
		// 			vk::ImageLayout::eTransferDstOptimal,
		// 			vk::ImageLayout::eShaderReadOnlyOptimal);
		// 	}
		// );

		reload();

		// Allocate and bind the descriptor set
		descriptor_set = device.allocateDescriptorSets({
			descriptor_pool, 1, &descriptor_set_layout,
		}).front();

		// vk::DescriptorImageInfo image_info {
		// 	sampler, density_image.view,
		// 	vk::ImageLayout::eShaderReadOnlyOptimal
		// };
		//
		// vk::WriteDescriptorSet write {
		// 	descriptor_set, 0, 0, 1,
		// 	vk::DescriptorType::eCombinedImageSampler,
		// 	&image_info, nullptr, nullptr
		// };
		//
		// device.updateDescriptorSets({ write }, {});
		littlevk::bind(device, descriptor_set, density_image, sampler);
	}

	size_t frame = 0;
	void render() {
		littlevk::SurfaceOperation op;
                op = littlevk::acquire_image(device, swapchain.swapchain, sync, frame);

		// Start empty render pass
		std::array <vk::ClearValue, 2> clear_values {
			vk::ClearColorValue { std::array <float, 4> { 0.0f, 0.0f, 0.0f, 1.0f } },
			vk::ClearDepthStencilValue { 1.0f, 0 }
		};

		vk::RenderPassBeginInfo render_pass_info {
			render_pass, framebuffers[op.index],
			vk::Rect2D { {}, window->extent },
			clear_values
		};

		// Record command buffer
		vk::CommandBuffer &cmd = command_buffers[frame];

		cmd.begin(vk::CommandBufferBeginInfo {});
		cmd.beginRenderPass(render_pass_info, vk::SubpassContents::eInline);

		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
			pipeline_layout, 0, { descriptor_set }, {});

		cmd.pushConstants <push_constants> (pipeline_layout,
			vk::ShaderStageFlagBits::eFragment, 0, constants);

		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		cmd.bindVertexBuffers(0, grid_vertices.buffer, { 0 });
		cmd.bindIndexBuffer(grid_indices.buffer, 0, vk::IndexType::eUint32);
		cmd.drawIndexed(6, 1, 0, 0, 0);

		cmd.endRenderPass();
		cmd.end();

		// Submit command buffer while signaling the semaphore
		vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;

		vk::SubmitInfo submit_info {
			1, &sync.image_available[frame],
			&wait_stage,
			1, &cmd,
			1, &sync.render_finished[frame]
		};

		graphics_queue.submit(submit_info, sync.in_flight[frame]);

                op = littlevk::present_image(present_queue, swapchain.swapchain, sync, op.index);
		frame = 1 - frame;
	}

	void reload() {
		// Reload density image
		littlevk::upload(device, density_staging, g_array.density);

		littlevk::submit_now(device, command_pool, graphics_queue,
			[&](const vk::CommandBuffer &cmd) {
				littlevk::transition(cmd, density_image,
					vk::ImageLayout::eUndefined,
					vk::ImageLayout::eTransferDstOptimal);

				littlevk::copy_buffer_to_image(cmd,
					density_image, density_staging,
					vk::ImageLayout::eTransferDstOptimal);

				littlevk::transition(cmd,
					density_image,
					vk::ImageLayout::eTransferDstOptimal,
					vk::ImageLayout::eShaderReadOnlyOptimal);
			}
		);
	}

	bool destroy() override {
		device.waitIdle();
		delete dal;
		return littlevk::Skeleton::destroy();
	}
} g_render;

int main()
{
	// Load Vulkan physical device
	auto predicate = [](const vk::PhysicalDevice &dev) {
		return littlevk::physical_device_able(dev,  {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		});
	};

	vk::PhysicalDevice phdev = littlevk::pick_physical_device(predicate);

	g_array.randomize();
	g_render.from(phdev);
	
	while (true) {
		glfwPollEvents();

		GLFWwindow *window = g_render.window->handle;
		if (glfwWindowShouldClose(window))
			break;

		// Viewport controls
		if (glfwGetKey(window, GLFW_KEY_LEFT))
			g_render.constants.center[0] += 0.01;
		else if (glfwGetKey(window, GLFW_KEY_RIGHT))
			g_render.constants.center[0] -= 0.01;

		if (glfwGetKey(window, GLFW_KEY_UP))
			g_render.constants.center[1] += 0.01;
		else if (glfwGetKey(window, GLFW_KEY_DOWN))
			g_render.constants.center[1] -= 0.01;

		// +/- for zoom controls
		if (glfwGetKey(window, GLFW_KEY_EQUAL)) {
			g_render.constants.extent[0] *= 1.01;
			g_render.constants.extent[1] *= 1.01;
		} else if (glfwGetKey(window, GLFW_KEY_MINUS)) {
			g_render.constants.extent[0] /= 1.01;
			g_render.constants.extent[1] /= 1.01;
		}

		// Constrain view
		float *ext = g_render.constants.extent;
		ext[0] = std::max(ext[0], 1.0f);
		ext[1] = std::max(ext[1], 1.0f);

		float *center = g_render.constants.center;
		// center[0] = std::max(0.0f, center[0]);
		// center[1] = std::max(0.0f, center[1]);

		g_array.add(0.2, 0.2, 0.1, 0.05);
		g_array.add(0.7, 0.6, -0.01, 0.3);
		g_array.add(0.3, 0.7, 0.05, 0.05);
		g_array.diffuse();

		g_render.reload();
		g_render.render();
	}

	g_render.destroy();

	return 0;
}

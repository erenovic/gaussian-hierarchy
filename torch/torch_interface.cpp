#include "torch_interface.h"
#include "../hierarchy_loader.h"
#include "../hierarchy_writer.h"
#include "../traversal.h"
#include "../runtime_switching.h"
#include "../PointbasedKdTreeGenerator.h"
#include "../ClusterMerger.h"
#include "../rotation_aligner.h"
#include "../writer.h"
#include "../common.h"
#include <cstring>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LoadHierarchy(std::string filename)
{
	HierarchyLoader loader;
	
	std::vector<Eigen::Vector3f> pos;
	std::vector<SHs> shs;
	std::vector<float> alphas;
	std::vector<Eigen::Vector3f> scales;
	std::vector<Eigen::Vector4f> rot;
	std::vector<Node> nodes;
	std::vector<Box> boxes;
	
	loader.load(filename.c_str(), pos, shs, alphas, scales, rot, nodes, boxes);
	
	int P = pos.size();
	
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	torch::Tensor pos_tensor = torch::from_blob(pos.data(), {P, 3}, options).clone();
	torch::Tensor shs_tensor = torch::from_blob(shs.data(), {P, 16, 3}, options).clone();
	torch::Tensor alpha_tensor = torch::from_blob(alphas.data(), {P, 1}, options).clone();
	torch::Tensor scale_tensor = torch::from_blob(scales.data(), {P, 3}, options).clone();
	torch::Tensor rot_tensor = torch::from_blob(rot.data(), {P, 4}, options).clone();
	
	int N = nodes.size();
	torch::TensorOptions intoptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
	
	torch::Tensor nodes_tensor = torch::from_blob(nodes.data(), {N, 7}, intoptions).clone();
	torch::Tensor box_tensor = torch::from_blob(boxes.data(), {N, 2, 4}, options).clone();
	
	return std::make_tuple(pos_tensor, shs_tensor, alpha_tensor, scale_tensor, rot_tensor, nodes_tensor, box_tensor);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BuildHierarchy(
torch::Tensor& positions,
torch::Tensor& shs,
torch::Tensor& opacities,
torch::Tensor& scales,
torch::Tensor& rotations)
{
	auto pos_cpu = positions.cpu().contiguous();
	auto shs_cpu = shs.cpu().contiguous();
	auto opacity_cpu = opacities.cpu().contiguous();
	auto scale_cpu = scales.cpu().contiguous();
	auto rot_cpu = rotations.cpu().contiguous();

	int P = pos_cpu.size(0);
	std::vector<Gaussian> gaussians;
	gaussians.reserve(P);

	auto pos_ptr = pos_cpu.data_ptr<float>();
	auto shs_ptr = shs_cpu.data_ptr<float>();
	auto opacity_ptr = opacity_cpu.data_ptr<float>();
	auto scale_ptr = scale_cpu.data_ptr<float>();
	auto rot_ptr = rot_cpu.data_ptr<float>();

	for (int i = 0; i < P; i++)
	{
		Gaussian g;
		g.position = Eigen::Vector3f(pos_ptr[i * 3 + 0], pos_ptr[i * 3 + 1], pos_ptr[i * 3 + 2]);
		g.opacity = opacity_ptr[i];
		g.scale = Eigen::Vector3f(scale_ptr[i * 3 + 0], scale_ptr[i * 3 + 1], scale_ptr[i * 3 + 2]);
		g.rotation = Eigen::Vector4f(rot_ptr[i * 4 + 0], rot_ptr[i * 4 + 1], rot_ptr[i * 4 + 2], rot_ptr[i * 4 + 3]);
		std::memcpy(g.shs.data(), shs_ptr + i * 48, sizeof(float) * 48);
		computeCovariance(g.scale, g.rotation, g.covariance);
		gaussians.push_back(g);
	}

	PointbasedKdTreeGenerator generator;
	auto root = generator.generate(gaussians);

	ClusterMerger merger;
	merger.merge(root, gaussians);

	RotationAligner::align(root, gaussians);

	std::vector<Eigen::Vector3f> out_positions;
	std::vector<Eigen::Vector4f> out_rotations;
	std::vector<Eigen::Vector3f> out_log_scales;
	std::vector<float> out_opacities;
	std::vector<SHs> out_shs;
	std::vector<Node> out_nodes;
	std::vector<Box> out_boxes;

	Writer::makeHierarchy(gaussians, root, out_positions, out_rotations, out_log_scales, out_opacities, out_shs, out_nodes, out_boxes);

	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	torch::TensorOptions intoptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

	torch::Tensor pos_tensor = torch::from_blob(out_positions.data(), {(int)out_positions.size(), 3}, options).clone();
	torch::Tensor shs_tensor = torch::from_blob(out_shs.data(), {(int)out_shs.size(), 16, 3}, options).clone();
	torch::Tensor alpha_tensor = torch::from_blob(out_opacities.data(), {(int)out_opacities.size(), 1}, options).clone();
	torch::Tensor log_scale_tensor = torch::from_blob(out_log_scales.data(), {(int)out_log_scales.size(), 3}, options).clone();
	torch::Tensor rot_tensor = torch::from_blob(out_rotations.data(), {(int)out_rotations.size(), 4}, options).clone();
	torch::Tensor nodes_tensor = torch::from_blob(out_nodes.data(), {(int)out_nodes.size(), 7}, intoptions).clone();
	torch::Tensor box_tensor = torch::from_blob(out_boxes.data(), {(int)out_boxes.size(), 2, 4}, options).clone();

	return std::make_tuple(pos_tensor, shs_tensor, alpha_tensor, log_scale_tensor, rot_tensor, nodes_tensor, box_tensor);
}

void WriteHierarchy(
					std::string filename,
					torch::Tensor& pos,
					torch::Tensor& shs,
					torch::Tensor& opacities,
					torch::Tensor& log_scales,
					torch::Tensor& rotations,
					torch::Tensor& nodes,
					torch::Tensor& boxes)
{
	HierarchyWriter writer;
	
	int allP = pos.size(0);
	int allN = nodes.size(0);
	
	writer.write(
		filename.c_str(),
		allP,
		allN,
		(Eigen::Vector3f*)pos.cpu().contiguous().data_ptr<float>(),
		(SHs*)shs.cpu().contiguous().data_ptr<float>(),
		opacities.cpu().contiguous().data_ptr<float>(),
		(Eigen::Vector3f*)log_scales.cpu().contiguous().data_ptr<float>(),
		(Eigen::Vector4f*)rotations.cpu().contiguous().data_ptr<float>(),
		(Node*)nodes.cpu().contiguous().data_ptr<int>(),
		(Box*)boxes.cpu().contiguous().data_ptr<float>()
	);
}

torch::Tensor
ExpandToTarget(torch::Tensor& nodes, int target)
{
	std::vector<int> indices = Traversal::expandToTarget((Node*)nodes.cpu().contiguous().data_ptr<int>(), target);
	torch::TensorOptions intoptions = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
	return torch::from_blob(indices.data(), {(int)indices.size()}, intoptions).clone();
}

int ExpandToSize(
torch::Tensor& nodes, 
torch::Tensor& boxes, 
float size, 
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& render_indices,
torch::Tensor& parent_indices,
torch::Tensor& nodes_for_render_indices)
{
	return Switching::expandToSize(
	nodes.size(0), 
	size,
	nodes.contiguous().data_ptr<int>(), 
	boxes.contiguous().data_ptr<float>(),
	viewpoint.contiguous().data_ptr<float>(),
	viewdir.data_ptr<float>()[0], viewdir.data_ptr<float>()[1], viewdir.data_ptr<float>()[2],
	render_indices.contiguous().data_ptr<int>(),
	nullptr,
	parent_indices.contiguous().data_ptr<int>(),
	nodes_for_render_indices.contiguous().data_ptr<int>());
}

void GetTsIndexed(
torch::Tensor& indices,
float size,
torch::Tensor& nodes,
torch::Tensor& boxes,
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& ts,
torch::Tensor& num_kids)
{
	Switching::getTsIndexed(
	indices.size(0),
	indices.contiguous().data_ptr<int>(),
	size,
	nodes.contiguous().data_ptr<int>(), 
	boxes.contiguous().data_ptr<float>(),
	viewpoint.data_ptr<float>()[0], viewpoint.data_ptr<float>()[1], viewpoint.data_ptr<float>()[2],
	viewdir.data_ptr<float>()[0], viewdir.data_ptr<float>()[1], viewdir.data_ptr<float>()[2],
	ts.contiguous().data_ptr<float>(),
	num_kids.contiguous().data_ptr<int>(), 0);
}

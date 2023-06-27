import numpy as np
import torch
import math
import os
from torch import optim
from torch import nn
import argparse

device = 'cuda' if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument('--obj_name', type=str, required=True, default='Object for which the symmetry will be calculated')
    parser.add_argument('--symmtype', default="reflectional", required=True, type=str, help='Whether to calculate rotational or reflectional symmetry')
    parser.add_argument('--prior', default=None, type=float, nargs='+', required=False, help='Symmetry prior from which to start the optimization')
    parser.add_argument('--optimize_center', action='store_true', help='Default symmetry center is the object center, use this to optimize the center')
    parser.add_argument('--n_rot', type=int, help='Number of rotational symmetries')
    args = parser.parse_args()
    return args


def mirror_mesh_cuda(mesh, center, sym):

    x1, y1, z1 = center[0], center[1], center[2]
    a, b, c = sym[0], sym[1], sym[2]            
    d = a*x1+b*y1+c*z1

    t = (d - (a * mesh[:, 0] + b * mesh[:, 1] + c * mesh[:, 2])) / (a * a + b * b + c * c)
    sym_x = 2 * a * t + mesh[:, 0]
    sym_y = 2 * b * t + mesh[:, 1]
    sym_z = 2 * c * t + mesh[:, 2]

    sym_mesh = torch.stack((sym_x, sym_y, sym_z), dim=1)
    return sym_mesh


def cal_adds_cuda(mesh, mesh_sym):
    N, _ = mesh.shape
    pd = mesh.view(1, N, 3).repeat(N, 1, 1)
    gt = mesh_sym.reshape(N, 1, 3).repeat(1, N, 1)

    dis = torch.norm(pd - gt, dim=2)
    mdis = torch.min(dis, dim=1)[0]
    return torch.mean(mdis)


def cal_adds_cuda_vec(mesh, mesh_sym):
    S, N, _ = mesh_sym.shape
    pd = mesh.view(1, N, 3).repeat(S, N, 1, 1)
    
    gt = mesh_sym.reshape(S, N, 1, 3).repeat(1, 1, N, 1)

    dis = torch.norm(pd - gt, dim=3)
    mdis = torch.min(dis, dim=1)[0]
    return torch.sum(torch.mean(mdis,dim=1))


def rotate(pt, v1, v2, theta):
    M = torch.zeros((4, 4))
    a, b, c = v1[0], v1[1], v1[2]

    p = ((v2 - v1) / torch.norm(v2 - v1)).reshape(-1)
    u, v, w = p[0],  p[1],  p[2]

    uu, uv, uw = u * u, u * v, u * w
    vv, vw, ww = v * v, v * w, w * w

    au, av, aw = a * u, a * v, a * w
    bu, bv, bw = b * u, b * v, b * w
    cu, cv, cw = c * u, c * v, c * w

    costheta = math.cos(theta * math.pi / 180)
    sintheta = math.sin(theta * math.pi / 180)

    M[0][0], M[0][1], M[0][2], M[0][3] = uu + (vv + ww) * costheta,  uv * (1 - costheta) + w * sintheta, uw * (1 - costheta) - v * sintheta, 0
    M[1][0], M[1][1], M[1][2], M[1][3] = uv * (1 - costheta) - w * sintheta, vv + (uu + ww) * costheta, vw * (1 - costheta) + u * sintheta, 0
    M[2][0], M[2][1], M[2][2], M[2][3] = uw * (1 - costheta) + v * sintheta, vw * (1 - costheta) - u * sintheta, ww + (uu + vv) * costheta, 0
    M[3][0], M[3][1], M[3][2], M[3][3] = (a * (vv + ww) - u * (bv + cw)) * (1 - costheta) + (bw - cv) * sintheta, (b * (uu + ww) - v * (au + cw)) * (1 - costheta) + (cu - aw) * sintheta, (c * (uu + vv) - w * (au + bv)) * (1 - costheta) + (av - bu) * sintheta, 0
    
    pt = torch.vstack((pt.T, torch.ones((1,pt.shape[0]))))
    V = M@pt

    return V.squeeze()[:3]


def rotate_cuda(pt, v1, v2, theta):
    M = torch.zeros((4, 4)).to(device)
    a, b, c = v1[0], v1[1], v1[2]

    p = ((v2 - v1) / torch.norm(v2 - v1)).reshape(-1)
    u, v, w = p[0],  p[1],  p[2]

    uu, uv, uw = u * u, u * v, u * w
    vv, vw, ww = v * v, v * w, w * w
    au, av, aw = a * u, a * v, a * w
    bu, bv, bw = b * u, b * v, b * w
    cu, cv, cw = c * u, c * v, c * w

    costheta = math.cos(theta * math.pi / 180)
    sintheta = math.sin(theta * math.pi / 180)

    M[0][0], M[0][1], M[0][2], M[0][3] = uu + (vv + ww) * costheta,  uv * (1 - costheta) + w * sintheta, uw * (1 - costheta) - v * sintheta, 0
    M[1][0], M[1][1], M[1][2], M[1][3] = uv * (1 - costheta) - w * sintheta, vv + (uu + ww) * costheta, vw * (1 - costheta) + u * sintheta, 0
    M[2][0], M[2][1], M[2][2], M[2][3] = uw * (1 - costheta) + v * sintheta, vw * (1 - costheta) - u * sintheta, ww + (uu + vv) * costheta, 0
    M[3][0], M[3][1], M[3][2], M[3][3] = (a * (vv + ww) - u * (bv + cw)) * (1 - costheta) + (bw - cv) * sintheta, (b * (uu + ww) - v * (au + cw)) * (1 - costheta) + (cu - aw) * sintheta, (c * (uu + vv) - w * (au + bv)) * (1 - costheta) + (av - bu) * sintheta, 0
    
    pt = torch.vstack((pt.T, torch.ones((1,pt.shape[0])).to(device)))
    V = M@pt
    
    return V.squeeze()[:3]


def rotate_cuda_vec(pt, v1, v2, theta):
    M = torch.zeros((4, 4,theta.shape[0])).to(device)
    a, b, c = v1[0], v1[1], v1[2]

    p = ((v2 - v1) / torch.norm(v2 - v1)).reshape(-1)
    u, v, w = p[0],  p[1],  p[2]

    uu, uv, uw = u * u, u * v, u * w
    vv, vw, ww = v * v, v * w, w * w
    au, av, aw = a * u, a * v, a * w
    bu, bv, bw = b * u, b * v, b * w
    cu, cv, cw = c * u, c * v, c * w

    costheta = torch.cos(theta * math.pi / 180)
    sintheta = torch.sin(theta * math.pi / 180)

    M[0][0], M[0][1], M[0][2], M[0][3] = uu + (vv + ww) * costheta,  uv * (1 - costheta) + w * sintheta, uw * (1 - costheta) - v * sintheta, 0
    M[1][0], M[1][1], M[1][2], M[1][3] = uv * (1 - costheta) - w * sintheta, vv + (uu + ww) * costheta, vw * (1 - costheta) + u * sintheta, 0
    M[2][0], M[2][1], M[2][2], M[2][3] = uw * (1 - costheta) + v * sintheta, vw * (1 - costheta) - u * sintheta, ww + (uu + vv) * costheta, 0
    M[3][0], M[3][1], M[3][2], M[3][3] = (a * (vv + ww) - u * (bv + cw)) * (1 - costheta) + (bw - cv) * sintheta, (b * (uu + ww) - v * (au + cw)) * (1 - costheta) + (cu - aw) * sintheta, (c * (uu + vv) - w * (au + bv)) * (1 - costheta) + (av - bu) * sintheta, 0
    
    M = M.permute(2,0,1)
    pt = torch.vstack((pt.T, torch.ones((1,pt.shape[0])).to(device)))
    
    V = torch.matmul(M,pt)

    return V.squeeze()[:,:3,:]


def rotate_mesh_cuda_vec(mesh, center, axis, angle):
    v1 = center + axis
    v2 = center - axis
    sym_mesh = rotate_cuda_vec(mesh, v1.squeeze(), v2.squeeze(), angle).permute(0,2,1)

    return sym_mesh


def rotate_mesh_cuda(mesh, center, axis, angle):
    v1 = center + axis
    v2 = center - axis
    sym_mesh = rotate_cuda(mesh, v1.squeeze(), v2.squeeze(), angle).T

    return sym_mesh


def optimize_rotation(mesh_pts, center, rand_normal, obj_name, optimize_center,n_rot=None):
    prior = rand_normal.detach().cpu().numpy()
    dir_rot = nn.Parameter((rand_normal/torch.norm(rand_normal,p=2)).to(device))

    if n_rot != None:
        rot_to_check = torch.tensor([int(i*(360/n_rot)) for i in range(1,n_rot)])
    else:
        rot_to_check = torch.tensor([30,60,90,120,150,180,210,240,270,300,330])

    if optimize_center:
        params = [center, dir_rot]
    else:
        params = [dir_rot]
        center = center.detach()
    optimizer = optim.Adam(params, lr=0.01)
    best_loss = np.inf
    best_dir_rot = None
    best_center = None
    for t in range(150):
        optimizer.zero_grad()
        if len(rot_to_check) > 1:
            sym_mesh = rotate_mesh_cuda_vec(mesh_pts, center, dir_rot, rot_to_check.to(device))
            loss = cal_adds_cuda_vec(mesh_pts, sym_mesh)
        else:
            sym_mesh = rotate_mesh_cuda(mesh_pts,center,dir_rot,rot_to_check[0])
            loss = cal_adds_cuda(mesh_pts,sym_mesh)

        if loss < best_loss:
            best_loss = loss.clone()
            best_dir_rot = dir_rot.clone()
            best_center = center.clone()
        print(t,loss)

        loss.backward()
        optimizer.step()

    rotated_mesh = rotate_mesh_cuda(mesh_pts,best_center,best_dir_rot,180)
    loss180 = cal_adds_cuda(mesh_pts,rotated_mesh)
    print(loss180)
    dir_rot = best_dir_rot.detach().cpu().numpy()
    center = best_center.detach().cpu().numpy()

    dir_rot = dir_rot/np.linalg.norm(dir_rot)
    root_path = 'symmetry/roational/'
    obj_path = os.path.join(root_path,f'{obj_name}')
    os.makedirs(obj_path, exist_ok=True)
    a = str(prior[0]).replace('.','-')
    b = str(prior[1]).replace('.','-')
    c = str(prior[2]).replace('.','-')
    with open(os.path.join(obj_path,f'symmetry_prior_{a}_{b}_{c}.txt'), 'w') as f:
        f.write(np.array2string(dir_rot, separator=','))
        if optimize_center:
            f.write('\n center: '+np.array2string(center, separator=','))
        f.write('\n loss: ' + str(loss180))

    print(dir_rot)


def optimize_reflection(mesh_pts, center, rand_normal, obj_name, optimize_center):
    prior = rand_normal.detach().cpu().numpy()
    
    normal_ref1 = nn.Parameter((rand_normal/torch.norm(rand_normal,p=2)).to(device))
    if optimize_center:
        print('optimize center')
        params = [normal_ref1, center]
    else:
        params = [normal_ref1]
        center = center.detach()
    lr = 0.1
    optimizer = optim.Adam(params, lr=lr)
    best_ref1 = None
    best_loss = np.inf
    best_center = None
    for t in range(500):
        optimizer.zero_grad()
        sym_mesh = mirror_mesh_cuda(mesh_pts, center, normal_ref1)
        loss = cal_adds_cuda(mesh_pts, sym_mesh)
        if loss < best_loss:
            best_loss = loss.clone()
            best_ref1 = normal_ref1.clone()
            best_center = center.clone()

        print(t, loss)
        loss.backward()
        optimizer.step()

    refl_mesh = mirror_mesh_cuda(mesh_pts, best_center, best_ref1)
    refl_loss = cal_adds_cuda(mesh_pts, refl_mesh)
    print(refl_loss)
    normal_ref1 = best_ref1.detach().cpu().numpy()

    center = best_center.detach().cpu().numpy()
    normal_ref1 = normal_ref1/np.linalg.norm(normal_ref1)
    print(normal_ref1)
    root_path = 'symmetry/reflectional/'
    obj_path = os.path.join(root_path,f'{obj_name}')
    os.makedirs(obj_path, exist_ok=True)
    d = '_center_' if optimize_center else ''
    a = str(prior[0]).replace('.','-')
    b = str(prior[1]).replace('.','-')
    c = str(prior[2]).replace('.','-')
    with open(os.path.join(obj_path,f'symmetry{d}_prior_{a}_{b}_{c}.txt'), 'w') as f:
        f.write(np.array2string(normal_ref1, separator=',')+'\n')
        f.write(f'Loss: {best_loss.detach().cpu().numpy():.4f}')
        if optimize_center:
            f.write('\ncenter: '+np.array2string(center, separator=','))
        f.write('\loss: ' + str(refl_loss))


def main():
    args = get_args()
    obj_name, symmtype, prior, optimize_center, n_rot = args.obj_name, args.symmtype, args.prior, args.optimize_center, args.n_rot
    root_path = "/path/to/ycb_kps/"
    models_root = "/path/to/models/"
    obj_center_path = os.path.join(root_path, obj_name + "_center.txt")
    mesh_path = os.path.join(models_root, obj_name, "points.xyz")

    center = np.loadtxt(obj_center_path, dtype=np.float32)
    center = nn.Parameter(torch.from_numpy(center).to(device))
    mesh_pts = torch.from_numpy(np.loadtxt(mesh_path, dtype=np.float32)).to(device)

    if prior is not None:
        rand_normal = torch.tensor(prior)
    else:
        rand_normal = torch.rand(3) 
    
    if symmtype == "reflectional":
        optimize_reflection(mesh_pts, center, rand_normal, obj_name, optimize_center)

    elif symmtype == "rotational":
        optimize_rotation(mesh_pts, center, rand_normal, obj_name, optimize_center, n_rot)

    else:
        raise Exception("Symmetry does not exist")


if __name__ == "__main__":
    main()

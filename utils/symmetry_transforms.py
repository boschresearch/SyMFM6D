import torch 
import numpy as np
import math


def mirror_mesh_cuda(mesh, center, sym):
    sym_mesh = mesh.clone()

    x1, y1, z1 = center[0], center[1], center[2]
    a, b, c = sym[0], sym[1], sym[2]            
    d = a*x1+b*y1+c*z1

    t = (d-(a*mesh[:,0]+b*mesh[:,1]+c*mesh[:,2]))/(a*a+b*b+c*c)
    sym_x = 2 * a * t + mesh[:,0]                      
    sym_y = 2 * b * t + mesh[:,1]
    sym_z = 2 * c * t + mesh[:,2]

    sym_mesh = torch.stack((sym_x,sym_y,sym_z),dim=1)
    return sym_mesh


def mirror_mesh_cuda_vec(mesh, center, sym):
    sym_mesh = mesh.clone()

    x1, y1, z1 = center[0], center[1], center[2]
    a, b, c = sym[...,0], sym[...,1], sym[...,2]            
    d = a*x1+b*y1+c*z1

    a,b,c,d = a.unsqueeze(1),b.unsqueeze(1),c.unsqueeze(1),d.unsqueeze(1)

    t = (d-(a*mesh[:,0]+b*mesh[:,1]+c*mesh[:,2]))/(a*a+b*b+c*c)
    sym_x = 2 * a * t + mesh[:,0]                      
    sym_y = 2 * b * t + mesh[:,1]
    sym_z = 2 * c * t + mesh[:,2]

    sym_mesh = torch.stack((sym_x,sym_y,sym_z),dim=-1)
    return sym_mesh


def mirror_mesh_np(mesh, center, sym):
    sym_mesh = mesh.copy()

    x1, y1, z1 = center[0], center[1], center[2]
    a, b, c = sym[0], sym[1], sym[2]            
    d = a*x1+b*y1+c*z1

    t = (d-(a*mesh[:,0]+b*mesh[:,1]+c*mesh[:,2]))/(a*a+b*b+c*c)
    sym_x = 2 * a * t + mesh[:,0]                      
    sym_y = 2 * b * t + mesh[:,1]
    sym_z = 2 * c * t + mesh[:,2]

    sym_mesh = np.stack((sym_x,sym_y,sym_z),axis=1)
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
    device = 'cuda' if torch.cuda.is_available() else "cpu"
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
    V  = M@pt
    
    return V.squeeze()[:3]


def rotate_np(pt, v1, v2, theta):
    M = np.zeros((4, 4))
    a, b, c = v1[0], v1[1], v1[2]

    p = ((v2 - v1) / np.linalg.norm(v2 - v1)).reshape(-1)
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
    
    pt = np.vstack((pt.T, np.ones((1,pt.shape[0]))))
    V = M@pt
    
    return V.squeeze()[:3]


def rotate_cuda_vec(pt, v1, v2, theta):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
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
    sym_mesh = mesh.clone()
    v1 = center + axis
    v2 = center - axis
    sym_mesh = rotate_cuda_vec(mesh, v1.squeeze(), v2.squeeze(), angle).permute(0,2,1)

    return sym_mesh


def rotate_mesh_cuda(mesh, center, axis, angle):
    sym_mesh = mesh.clone()
    v1 = center + axis
    v2 = center - axis
    sym_mesh = rotate_cuda(mesh, v1.squeeze(), v2.squeeze(), angle).T

    return sym_mesh


def reflect_kps(kpts, obj):
    pass

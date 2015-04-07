local cunntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}

--e.g.: th -lcunn -e "nn.testcuda{'copies'}"

function cunntest.SpatialConvolutionCUDA_forward_batch()
   local bs = 32
   local from = 4 * math.random(1,4)
   local to = 32
   local ki = math.random(3,15)
   local kj = ki
   local si = math.random(1,2)
   local sj = si
   local outi = math.random(1,64)
   local outj = outi
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialConvolutionCUDA.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi, sj, si)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:resize(bs,from*ini*inj):t():contiguous():resize(from,ini,inj,bs):cuda()
   local gconv = nn.SpatialConvolutionCUDA(from,to,ki,kj,si,sj):cuda()

   local weight = sconv.weight:clone()
   weight:resize(to, from*ki*kj)
   weight = weight:t():contiguous()
   weight:resize(from, kj, ki, to)
   gconv.weight:copy(weight)
   gconv.bias:copy(sconv.bias)

   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   rescuda = rescuda:resize(to*outi*outj,bs):t():contiguous():resize(bs,to,outi,outj):float()

   local error = rescuda - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialConvolutionCUDA_backward_batch()
   local bs = 32
   local from = 4 * math.random(1,4)
   local to = 32
   local ki = math.random(5,11)
   local kj = ki
   local si = math.random(1,2)
   local sj = si
   local outi = math.random(4,12)
   local outj = outi
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialConvolution.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialConvolution(from,to,ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   input = input:resize(bs,from*ini*inj):t():contiguous():resize(from,ini,inj,bs):cuda()
   gradOutput = gradOutput:resize(bs,to*outi*outj):t():contiguous():resize(to,outi,outj,bs):cuda()
   local gconv = nn.SpatialConvolutionCUDA(from,to,ki,kj,si,sj):cuda()

   local weight = sconv.weight:clone()
   weight:resize(to, from*ki*kj)
   weight = weight:t():contiguous()
   weight:resize(from, kj, ki, to)
   gconv.weight:copy(weight)
   gconv.bias:copy(sconv.bias)

   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real

   rescuda = rescuda:resize(from*ini*inj,bs):t():contiguous():resize(bs,from,ini,inj)
   weightcuda = weightcuda:resize(from*ki*kj, to):t():contiguous():resize(to, from, ki, kj)

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cunntest.SpatialMaxPoolingCUDA_forward_batch()
   local bs = 32
   local from = 16 * math.random(1,3)
   local to = from
   local ki = math.random(2,4)
   local kj = ki
   local si = ki
   local sj = kj
   local outi = math.random(16,32)
   local outj = outi
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialMaxPoolingCUDA.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:resize(bs,from*ini*inj):t():contiguous():resize(from,ini,inj,bs):cuda()
   local gconv = nn.SpatialMaxPoolingCUDA(ki,kj,si,sj):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   rescuda = rescuda:resize(to*outi*outj,bs):t():contiguous():resize(bs,to,outi,outj):float()

   local error = rescuda - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialMaxPoolingCUDA_backward_batch()
   local bs = 32
   local from = 16 * math.random(1,3)
   local to = from
   local ki = math.random(2,4)
   local kj = ki
   local si = ki
   local sj = kj
   local outi = math.random(16,32)
   local outj = outi
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialMaxPoolingCUDA.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:resize(bs,from*ini*inj):t():contiguous():resize(from,ini,inj,bs):cuda()
   gradOutput = gradOutput:resize(bs,to*outi*outj):t():contiguous():resize(to,outi,outj,bs):cuda()
   local gconv = nn.SpatialMaxPoolingCUDA(ki,kj,si,sj):cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   rescuda = rescuda:resize(from*ini*inj,bs):t():contiguous():resize(bs,from,ini,inj)

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end


nloop = n_loop or nloop
local oldtype = torch.getdefaulttensortype()
torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
mytester = torch.Tester()
mytester:add(cunntest)
mytester:run(tests)
torch.setdefaulttensortype(oldtype)

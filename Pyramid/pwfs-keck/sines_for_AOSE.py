import numpy as np

import inspect
import copy
import warnings

class Grid(object):
	'''A set of points on some coordinate system.

	Parameters
	----------
	coords : CoordsBase
		The actual definition of the coordinate values.
	weights : array_like or None
		The interval size, area, volume or hypervolume of each point, depending on the number of dimensions.
		If this is None (default), the weights will be attempted to be calculated on the fly when needed.

	Attributes
	----------
	coords
		The coordinate values for each dimension.
	'''

	_coordinate_system = 'none'
	_coordinate_system_transformations = {}

	def __init__(self, coords, weights=None):
		self.coords = coords
		self.weights = weights

	def copy(self):
		'''Create a copy.
		'''
		return copy.deepcopy(self)

	def subset(self, criterium):
		'''Construct a subset of the current sampling, based on `criterium`.

		Parameters
		----------
		criterium : function or array_like
			The criterium used to select points. A function will be evaluated for every point. Otherwise, this must be a boolean array of integer array, used for slicing the points.
		'''
		if hasattr(criterium, '__call__'):
			indices = criterium(self) != 0
		else:
			indices = criterium
		new_coords = [c[indices] for c in self.coords]
		new_weights = self.weights[indices]
		return self.__class__(UnstructuredCoords(new_coords), new_weights)

	@property
	def ndim(self):
		'''The number of dimensions.
		'''
		return len(self.coords)

	@property
	def size(self):
		'''The number of points in this grid.
		'''
		return self.coords.size

	@property
	def dims(self):
		'''The number of elements in each dimension for a separated grid.

		Raises
		------
		ValueError
			If the grid is not separated.
		'''
		if not self.is_separated:
			raise ValueError('A non-separated grid does not have dims.')
		return self.coords.dims

	@property
	def shape(self):
		'''The shape of a reshaped ``numpy.ndarray`` using this grid.

		Raises
		------
		ValueError
			If the grid is not separated.
		'''
		if not self.is_separated:
			raise ValueError('A non-separated grid does not have a shape.')
		return self.coords.shape

	@property
	def delta(self):
		'''The spacing between points in regularly-spaced grid.

		Raises
		------
		ValueError
			If the grid is not regular.
		'''
		if not self.is_regular:
			raise ValueError('A non-regular grid does not have a delta.')
		return self.coords.delta

	@property
	def zero(self):
		'''The zero point of a regularly-spaced grid.

		Raises
		------
		ValueError
			If the grid is not regular.
		'''
		if not self.is_regular:
			raise ValueError('A non-regular grid does not have a zero.')
		return self.coords.zero

	@property
	def separated_coords(self):
		'''A list of coordinates for each dimension in a separated grid.

		Raises
		------
		ValueError
			If the grid is not separated.
		'''
		if not self.is_separated:
			raise ValueError('A non-separated grid does not have separated coordinates.')
		return self.coords.separated_coords

	@property
	def regular_coords(self):
		'''The tuple (delta, dims, zero) for a regularly-spaced grid.

		Raises
		------
		ValueError
			If the grid is not regular.
		'''
		if not self.is_regular:
			raise ValueError('A non-regular grid does not have regular coordinates.')
		return self.coords.regular_coords

	@property
	def weights(self):
		'''The interval size, area, volume or hypervolume of each point, depending on the number of dimensions.

		The weights are attempted to be calculated on the fly if not set. If this fails, a warning is emitted and all points will be given an equal weight of one.
		'''
		if self._weights is None:
			self._weights = self.__class__._get_automatic_weights(self.coords)

			if self._weights is None:
				self._weights = 1
				warnings.warn('No automatic weights could be calculated for this grid.', stacklevel=2)

		if np.isscalar(self._weights):
			return np.ones(self.size) * self._weights
		else:
			return self._weights

	@weights.setter
	def weights(self, weights):
		self._weights = weights

	@property
	def points(self):
		'''A list of points of this grid.

		This can be used for easier iteration::

			for p in grid.points:
				print(p)
		'''
		return np.array(self.coords).T

	@property
	def is_separated(self):
		'''True if the grid is separated, False otherwise.
		'''
		return self.coords.is_separated

	@property
	def is_regular(self):
		'''True if the grid is regularly-spaced, False otherwise.
		'''
		return self.coords.is_regular

	def is_(self, system):
		'''Check if the coordinate system is `system`.

		Parameters
		----------
		system : str
			The name of the coordinate system to check for.

		Returns
		-------
		bool
			If the coordinate system of the grid is equal to `system`.
		'''
		return system == self._coordinate_system

	def as_(self, system):
		'''Convert the grid to the new coordinate system `system`.

		If the grid is already in the right coordinate system, this function doesn't do anything.

		Parameters
		----------
		system : str
			The name of the coordinate system to check for.

		Returns
		-------
		Grid
			A new :class:`Grid` in the required coordinate system.

		Raises
		------
		ValueError
			If the conversion to the coordinate system `system` isn't known.
		'''
		if self.is_(system):
			return self
		else:
			return Grid._coordinate_system_transformations[self._coordinate_system][system](self)

	@staticmethod
	def _add_coordinate_system_transformation(source, dest, func):
		if source in Grid._coordinate_system_transformations:
			Grid._coordinate_system_transformations[source][dest] = func
		else:
			Grid._coordinate_system_transformations[source] = {dest: func}

	def __getitem__(self, i):
		'''The `i`-th point in this grid.
		'''
		return self.points[i]

	def scale(self, scale):
		'''Scale the grid in-place.

		Parameters
		----------
		scale : array_like
			The factor with which to scale the grid.

		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		raise NotImplementedError()

	def scaled(self, scale):
		'''A scaled copy of this grid.

		Parameters
		----------
		scale : array_like
			The factor with which to scale the grid.

		Returns
		-------
		Grid
			The scaled grid.
		'''
		grid = self.copy()
		grid.scale(scale)
		return grid

	def shift(self, shift):
		'''Shift the grid in-place.

		Parameters
		----------
		shift : array_like
			The amount with which to shift the grid.

		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		raise NotImplementedError()

	def shifted(self, shift):
		'''A shifted copy of this grid.

		Parameters
		----------
		shift : array_like
			The amount with which to shift the grid.

		Returns
		-------
		Grid
			The scaled grid.
		'''
		grid = self.copy()
		grid.shift(shift)
		return grid

	def reverse(self):
		'''Reverse the order of the points in-place.

		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		self.coords.reverse()
		return self

	def reversed(self):
		'''Make a copy of the grid with the order of the points reversed.

		Returns
		-------
		Grid
			The reversed grid.
		'''
		grid = self.copy()
		grid.reverse()
		return grid

	@staticmethod
	def _get_automatic_weights(coords):
		raise NotImplementedError()

	def __str__(self):
		return str(self.__class__) + '(' + str(self.coords.__class__) + ')'

	def closest_to(self, p):
		'''Get the index of the point closest to point `p`.

		Point `p` is assumed to have the same coordinate system as the grid itself.

		Parameters
		----------
		p : array_like
			The point at which to search for.

		Returns
		-------
		int
			The index of the closest point.
		'''
		rel_points = self.points - np.array(p) * np.ones(self.ndim)
		return np.argmin(np.sum(rel_points**2, axis=-1))

class CartesianGrid(Grid):
	'''A grid representing a N-dimensional Cartesian coordinate system.
	'''

	_coordinate_system = 'cartesian'

	@property
	def x(self):
		'''The x-coordinate (dimension 0).
		'''
		return self.coords[0]

	@property
	def y(self):
		'''The y-coordinate (dimension 1).
		'''
		return self.coords[1]

	@property
	def z(self):
		'''The z-coordinate (dimension 2).
		'''
		return self.coords[2]

	@property
	def w(self):
		'''The w-coordinate (dimension 3).
		'''
		return self.coords[3]

	def scale(self, scale):
		'''Scale the grid in-place.

		Parameters
		----------
		scale : array_like
			The factor with which to scale the grid.

		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		self.weights *= np.abs(scale)**self.ndim
		self.coords *= scale
		return self

	def shift(self, shift):
		'''Shift the grid in-place.

		Parameters
		----------
		shift : array_like
			The amount with which to shift the grid.

		Returns
		-------
		Grid
			Itself to allow for chaining these transformations.
		'''
		self.coords += shift
		return self

	@staticmethod
	def _get_automatic_weights(coords):
		if coords.is_regular:
			return np.prod(coords.delta)
		elif coords.is_separated:
			weights = []
			for i in range(len(coords)):
				x = coords.separated_coords[i]
				w = (x[2:] - x[:-2]) / 2.
				w = np.concatenate(([x[1] - x[0]], w, [x[-1] - x[-2]]))
				weights.append(w)

			return np.multiply.reduce(np.ix_(*weights[::-1])).ravel()

class CoordsBase(object):
	'''Base class for coordinates.
	'''

	def copy(self):
		'''Make a copy.
		'''
		return copy.deepcopy(self)

	def __add__(self, b):
		'''Add `b` to the coordinates separately and return the result.
		'''
		res = self.copy()
		res += b
		return res

	def __iadd__(self, b):
		'''Add `b` to the coordinates separately in-place.
		'''
		raise NotImplementedError()

	def __radd__(self, b):
		'''Add `b` to the coordinates separately and return the result.
		'''
		return self + b

	def __sub__(self, b):
		'''Subtract `b` from the coordinates separately and return the result.
		'''
		return self + (-b)

	def __isub__(self, b):
		'''Subtract `b` from the coordinates separately in-place.
		'''
		self += (-b)
		return self

	def __mul__(self, f):
		'''Multiply each coordinate with `f` separately and return the result.
		'''
		res = self.copy()
		res *= f
		return res

	def __rmul__(self, f):
		'''Multiply each coordinate with `f` separately and return the result.
		'''
		return self * f

	def __imul__(self, f):
		'''Multiply each coordinate with `f` separately in-place.
		'''
		raise NotImplementedError()

	def __div__(self, f):
		'''Divide each coordinate with `f` separately and return the result.
		'''
		return self * (1./f)

	def __idiv__(self, f):
		'''Divide each coordinate with `f` separately in-place.
		'''
		self *= (1./f)
		return self

	def __getitem__(self, i):
		'''The `i`-th point for these coordinates.
		'''
		raise NotImplementedError()

	@property
	def is_separated(self):
		'''True if the coordinates are separated, False otherwise.
		'''
		return hasattr(self, 'separated_coords')

	@property
	def is_regular(self):
		'''True if the coordinates are regularly-spaced, False otherwise.
		'''
		return hasattr(self, 'regular_coords')

	def reverse(self):
		'''Reverse the ordering of points in-place.
		'''
		raise NotImplementedError()

	@property
	def size(self):
		'''The number of points.
		'''
		raise NotImplementedError()

	def __len__(self):
		'''The number of dimensions.
		'''
		raise NotImplementedError()

class OpticalElement(object):
	def __call__(self, wavefront):
		return self.forward(wavefront)

	def forward(self, wavefront):
		raise NotImplementedError()

	def backward(self, wavefront):
		raise NotImplementedError()

	def get_transformation_matrix_forward(self, wavelength=1):
		raise NotImplementedError()

	def get_transformation_matrix_backward(self, wavelength=1):
		raise NotImplementedError()

	def get_instance(self, input_grid, wavelength):
		return self

def make_polychromatic(evaluated_arguments=None, num_in_cache=50):
	def decorator(optical_element):
		class PolychromaticOpticalElement(OpticalElement):
			def __init__(self, *args, **kwargs):
				self.wavelengths = []
				self.monochromatic_optical_elements = []
				self.monochromatic_args = args
				self.monochromatic_kwargs = kwargs

				if evaluated_arguments is not None:
					init = optical_element.__init__
					if hasattr(inspect, 'signature'):
						# Python 3
						monochromatic_arg_names = list(inspect.signature(init).parameters.keys())[1:]
					else:
						# Python 2
						monochromatic_arg_names = inspect.getargspec(init).args

					self.evaluate_arg = [m in evaluated_arguments for m in monochromatic_arg_names]

			def get_instance(self, input_grid, wavelength):
				if self.wavelengths:
					i = np.argmin(np.abs(wavelength - np.array(self.wavelengths)))
					wavelength_closest = self.wavelengths[i]

					delta_wavelength = np.abs(wavelength - wavelength_closest)
					if (delta_wavelength / wavelength) < 1e-6:
						return self.monochromatic_optical_elements[i]

				if evaluated_arguments is not None:
					args = list(self.monochromatic_args)
					kwargs = dict(self.monochromatic_kwargs)

					for i, (arg, ev) in enumerate(zip(args, self.evaluate_arg)):
						if ev and callable(arg):
							args[i] = arg(wavelength)

					for key, val in kwargs.items():
						if key in evaluated_arguments and callable(val):
							kwargs[key] = val(wavelength)

					elem = optical_element(*args, wavelength=wavelength, **kwargs)
				else:
					elem = optical_element(*self.monochromatic_args, wavelength=wavelength, **self.monochromatic_kwargs)

				self.wavelengths.append(wavelength)
				self.monochromatic_optical_elements.append(elem)

				if len(self.wavelengths) > num_in_cache:
					self.wavelengths.pop(0)
					self.monochromatic_optical_elements.pop(0)

				return elem

			def forward(self, wavefront):
				return self.get_instance(wavefront.electric_field.grid, wavefront.wavelength).forward(wavefront)

			def backward(self, wavefront):
				return self.get_instance(wavefront.electric_field.grid, wavefront.wavelength).backward(wavefront)

			def get_transformation_matrix_forward(self, input_grid, wavelength):
				return self.get_instance(input_grid, wavelength).get_transformation_matrix_forward(input_grid, wavelength)

			def get_transformation_matrix_backward(self, input_grid, wavelength):
				return self.get_instance(input_grid, wavelength).get_transformation_matrix_backward(input_grid, wavelength)

		return PolychromaticOpticalElement
	return decorator

class SurfaceApodizerMonochromatic(OpticalElement):
	def __init__(self, surface, refractive_index, wavelength):
		self.surface = surface
		self.refractive_index = refractive_index

	def forward(self, wavefront):
		opd = (self.refractive_index - 1) * self.surface

		wf = wavefront.copy()
		wf.electric_field *= np.exp(1j * opd * wf.wavenumber)

		return wf

	def backward(self, wavefront):
		opd = (self.refractive_index - 1) * self.surface

		wf = wavefront.copy()
		wf.electric_field *= np.exp(-1j * opd * wf.wavenumber)

		return wf

SurfaceApodizer = make_polychromatic(["surface", "refractive_index"])(SurfaceApodizerMonochromatic)

class Field(np.ndarray):
	'''The value of some physical quantity for each point in some coordinate system.

	Parameters
	----------
	arr : array_like
		An array of values or tensors for each point in the :class:`Grid`.
	grid : Grid
		The corresponding :class:`Grid` on which the values are set.

	Attributes
	----------
	grid : Grid
		The grid on which the values are defined.

	'''
	def __new__(cls, arr, grid):
		obj = np.asarray(arr).view(cls)
		obj.grid = grid
		return obj

	def __array_finalize__(self, obj):
		if obj is None:
			return
		self.grid = getattr(obj, 'grid', None)

	@property
	def tensor_order(self):
		'''The order of the tensor of the field.
		'''
		return self.ndim - 1

	@property
	def tensor_shape(self):
		'''The shape of the tensor of the field.
		'''
		return np.array(self.shape)[:-1]

	@property
	def is_scalar_field(self):
		'''True if this field is a scalar field (ie. a tensor order of 0), False otherwise.
		'''
		return self.tensor_order == 0

	@property
	def is_vector_field(self):
		'''True if this field is a vector field (ie. a tensor order of 1), False otherwise.
		'''
		return self.tensor_order == 1

	@property
	def is_valid_field(self):
		'''True if the field corresponds with its grid.
		'''
		return self.shape[-1] == self.grid.size

	@property
	def shaped(self):
		'''The reshaped version of this field.

		Raises
		------
		ValueError
			If this field isn't separated, no reshaped version can be made.
		'''
		if not self.grid.is_separated:
			raise ValueError('This field doesn\'t have a shape.')

		if self.tensor_order > 0:
			new_shape = np.concatenate([np.array(self.shape)[:-1], self.grid.shape])
			return self.reshape(new_shape)

		return self.reshape(self.grid.shape)

	def at(self, p):
		'''The value of this field closest to point p.

		Parameters
		----------
		p : array_like
			The point at which the closest value should be returned.

		Returns
		-------
		array_like
			The value, potentially tensor, closest to point p.

		'''
		i = self.grid.closest_to(p)
		return self[...,i]

class MonochromaticPropagator(OpticalElement):
	def __init__(self, wavelength):
		self.wavelength = wavelength

def make_propagator(monochromatic_propagator):
	class Propagator(OpticalElement):
		def __init__(self, *args, **kwargs):
			self.wavelengths = []
			self.monochromatic_propagators = []
			self.monochromatic_args = args
			self.monochromatic_kwargs = kwargs

		def get_monochromatic_propagator(self, wavelength):
			if len(self.wavelengths) > 0:
				i = np.argmin(np.abs(wavelength - np.array(self.wavelengths)))
				wavelength_closest = self.wavelengths[i]

				delta_wavelength = np.abs(wavelength - wavelength_closest)
				if (delta_wavelength / wavelength) < 1e-6:
					return self.monochromatic_propagators[i]

			m = monochromatic_propagator(*self.monochromatic_args, wavelength=wavelength, **self.monochromatic_kwargs)

			self.wavelengths.append(wavelength)
			self.monochromatic_propagators.append(m)

			if len(self.monochromatic_propagators) > 50:
				self.wavelengths.pop(0)
				self.monochromatic_propagators.pop(0)


			return m

		def __call__(self, wavefront):
			return self.forward(wavefront)

		def forward(self, wavefront):
			return self.get_monochromatic_propagator(wavefront.wavelength).forward(wavefront)

		def backward(self, wavefront):
			return self.get_monochromatic_propagator(wavefront.wavelength).backward(wavefront)

		def get_transformation_matrix_forward(self, wavelength=1):
			return self.get_monochromatic_propagator(wavelength).get_transformation_matrix_forward()

		def get_transformation_matrix_backward(self, wavelength=1):
			return self.get_monochromatic_propagator(wavelength).get_transformation_matrix_backward()

	return Propagator

class FraunhoferPropagatorMonochromatic(MonochromaticPropagator):
	def __init__(self, input_grid, output_grid, wavelength_0=1, focal_length=1, wavelength=1):
		if focal_length is None:
			f_lambda_ref = 1
		else:
			f_lambda_ref = wavelength_0 * focal_length

		f_lambda = f_lambda_ref * (wavelength / wavelength_0)
		self.uv_grid = output_grid.scaled(2*np.pi / f_lambda)
		self.fourier_transform = make_fourier_transform(input_grid, self.uv_grid)
		self.output_grid = output_grid

		# Intrinsic to Fraunhofer propagation
		self.norm_factor = 1 / (1j * f_lambda)
		self.input_grid = input_grid

	def forward(self, wavefront):
		U_new = self.fourier_transform.forward(wavefront.electric_field) * self.norm_factor
		return Wavefront(Field(U_new, self.output_grid), wavefront.wavelength)

	def backward(self, wavefront):
		U_new = self.fourier_transform.backward(wavefront.electric_field) / self.norm_factor
		return Wavefront(Field(U_new, self.input_grid), wavefront.wavelength)

	def get_transformation_matrix_forward(self, wavelength=1):
		# Ignore input wavelength and just use the internal one.
		return self.fourier_transform.get_transformation_matrix_forward() * self.norm_factor

	def get_transformation_matrix_backward(self, wavelength=1):
		# Ignore input wavelength and just use the internal one.
		return self.fourier_transform.get_transformation_matrix_backward() / self.norm_factor

FraunhoferPropagator = make_propagator(FraunhoferPropagatorMonochromatic)

def make_fft_grid(input_grid, q=1, fov=1):
	q = np.ones(input_grid.ndim, dtype='float') * q
	fov = np.ones(input_grid.ndim, dtype='float') * fov

	delta = (2*np.pi / (input_grid.delta * input_grid.dims)) / q
	dims = (input_grid.dims * fov * q).astype('int')
	zero = delta * (-dims/2 + np.mod(dims, 2) * 0.5)

	return CartesianGrid(RegularCoords(delta, dims, zero))

def multiplex_for_tensor_fields(func):
	'''A decorator for automatically multiplexing a function over the tensor directions.

	This function is used internally for simplifying the implementation of the Fourier transforms.

	Parameters
	----------
	func : function
		The function to multiplex. This function gets called for each of the tensor elements.
	'''
	def inner(self, field):
		if field.is_scalar_field:
			return func(self, field)
		else:
			f = field.reshape((-1,field.grid.size))
			res = [func(self, ff) for ff in f]
			new_shape = np.concatenate((field.tensor_shape, [-1]))
			return Field(np.array(res).reshape(new_shape), res[0].grid)

	return inner

class FourierTransform(object):
	def forward(self, field):
		raise NotImplementedError()

	def backward(self, field):
		raise NotImplementedError()

	def get_transformation_matrix_forward(self):
		coords_in = self.input_grid.as_('cartesian').coords
		coords_out = self.output_grid.as_('cartesian').coords

		A = np.exp(-1j * np.dot(np.array(coords_out).T, coords_in))
		A *= self.input_grid.weights

		return A

	def get_transformation_matrix_backward(self):
		coords_in = self.input_grid.as_('cartesian').coords
		coords_out = self.output_grid.as_('cartesian').coords

		A = np.exp(1j * np.dot(np.array(coords_in).T, coords_out))
		A *= self.output_grid.weights
		A /= (2*np.pi)**self.input_grid.ndim

		return A

class FastFourierTransform(FourierTransform):
	def __init__(self, input_grid, q=1, fov=1, shift=0):
		self.input_grid = input_grid

		self.shape_in = input_grid.shape
		self.weights = input_grid.weights
		self.size = input_grid.size
		self.ndim = input_grid.ndim

		self.output_grid = make_fft_grid(input_grid, q, fov).shifted(shift)

		self.shape_out = self.output_grid.shape
		self.internal_shape = (self.shape_in * q).astype('int')

		cutout_start = (self.internal_shape / 2.).astype('int') - (self.shape_in / 2.).astype('int')
		cutout_end = cutout_start + self.shape_in
		self.cutout_input = [slice(start, end) for start, end in zip(cutout_start, cutout_end)]

		cutout_start = (self.internal_shape / 2.).astype('int') - (self.shape_out / 2.).astype('int')
		cutout_end = cutout_start + self.shape_out
		self.cutout_output = [slice(start, end) for start, end in zip(cutout_start, cutout_end)]

		center = input_grid.zero + input_grid.delta * (np.array(input_grid.dims) // 2)
		if np.allclose(center, 0):
			self.shift_input = 1
		else:
			self.shift_input = np.exp(-1j * np.dot(center, self.output_grid.coords))
			self.shift_input /= np.fft.ifftshift(self.shift_input.reshape(self.shape_out)).ravel()[0] # No piston shift (remove central shift phase)

		shift = np.ones(self.input_grid.ndim) * shift
		if np.allclose(shift, 0):
			self.shift_output = 1
		else:
			self.shift_output = np.exp(-1j * np.dot(shift, self.input_grid.coords))

	@multiplex_for_tensor_fields
	def forward(self, field):
		f = np.zeros(self.internal_shape, dtype='complex')
		f[self.cutout_input] = (field.ravel() * self.weights * self.shift_output).reshape(self.shape_in)
		res = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(f)))
		res = res[self.cutout_output].ravel() * self.shift_input

		return Field(res, self.output_grid)

	@multiplex_for_tensor_fields
	def backward(self, field):
		f = np.zeros(self.internal_shape, dtype='complex')
		f[self.cutout_output] = (field.ravel() / self.shift_input).reshape(self.shape_out)
		res = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(f)))
		res = res[self.cutout_input].ravel() / self.weights / self.shift_output

		return Field(res, self.input_grid)

class MatrixFourierTransform(FourierTransform):
	def __init__(self, input_grid, output_grid):
		# Check input grid assumptions
		if not input_grid.is_separated or not input_grid.is_('cartesian'):
			raise ValueError('The input_grid must be separable in cartesian coordinates.')
		if not output_grid.is_separated or not output_grid.is_('cartesian'):
			raise ValueError('The output_grid must be separable in cartesian coordinates.')
		if not input_grid.ndim in [1,2]:
			raise ValueError('The input_grid must be one- or two-dimensional.')
		if input_grid.ndim != output_grid.ndim:
			raise ValueError('The input_grid must have the same dimensions as the output_grid.')

		self.input_grid = input_grid

		self.shape = input_grid.shape
		self.weights = input_grid.weights.ravel()
		self.output_grid = output_grid
		self.ndim = input_grid.ndim

		self.output_grid = output_grid

		if self.ndim == 1:
			self.M = np.exp(-1j * np.dot(output_grid.x[:,np.newaxis], input_grid.x[np.newaxis,:]))
		elif self.ndim == 2:
			self.M1 = np.exp(-1j * np.dot(output_grid.coords.separated_coords[1][:,np.newaxis], input_grid.coords.separated_coords[1][np.newaxis,:]))
			self.M2 = np.exp(-1j * np.dot(output_grid.coords.separated_coords[0][:,np.newaxis], input_grid.coords.separated_coords[0][np.newaxis,:])).T

	@multiplex_for_tensor_fields
	def forward(self, field):
		if self.ndim == 1:
			f = field.ravel() * self.weights
			res = np.dot(self.M, f)
		elif self.ndim == 2:
			f = (field.ravel() * self.weights).reshape(self.shape)
			res = np.dot(np.dot(self.M1, f), self.M2).ravel()

		return Field(res, self.output_grid)

	@multiplex_for_tensor_fields
	def backward(self, field):
		if self.ndim == 1:
			f = field.ravel() * self.output_grid.weights
			res = np.dot(self.M.conj().T, f)
		elif self.ndim == 2:
			f = (field.ravel() * self.output_grid.weights).reshape(self.output_grid.shape)
			res = np.dot(np.dot(self.M1.conj().T, f), self.M2.conj().T).ravel()

		res /= (2*np.pi)**self.ndim

		return Field(res, self.input_grid)

class NaiveFourierTransform(FourierTransform):
	def __init__(self, input_grid, output_grid):
		self.input_grid = input_grid
		self.output_grid = output_grid

	@multiplex_for_tensor_fields
	def forward(self, field):
		T = self.get_transformation_matrix_forward()
		res = T.dot(field.ravel())

		return Field(res, self.output_grid)

	@multiplex_for_tensor_fields
	def backward(self, field):
		T = self.get_transformation_matrix_backward()
		res = T.dot(field.ravel())

		return Field(res, self.input_grid)

def make_fourier_transform(input_grid, output_grid=None, q=1, fov=1, planner='estimate'):
	if output_grid is None:
		# Choose between FFT and MFT
		if not (input_grid.is_regular and input_grid.is_('cartesian')):
			raise ValueError('For non-regular non-cartesian Grids, a Fourier transform is required to have an output_grid.')

		if input_grid.ndim not in [1,2]:
			method = 'fft'
		else:
			output_grid = make_fft_grid(input_grid, q, fov)

			if planner == 'estimate':
				# Estimate analytically from complexities
				N_in = input_grid.shape * q
				N_out = output_grid.shape

				if input_grid.ndim == 1:
					fft = 4 * N_in[0] * np.log2(N_in)
					mft = 4 * input_grid.size * N_out[0]
				else:
					fft = 4 * np.prod(N_in) * np.log2(np.prod(N_in))
					mft = 4 * (np.prod(input_grid.shape) * N_out[1] + np.prod(N_out) * input_grid.shape[0])
				if fft > mft:
					method = 'mft'
				else:
					method = 'fft'
			elif planner == 'measure':
				# Measure directly
				fft = FastFourierTransform(input_grid, q, fov)
				mft = MatrixFourierTransform(input_grid, output_grid)

				a = np.zeros(input_grid.size, dtype='complex')
				fft_time = time_it(lambda: fft.forward(a))
				mft_time = time_it(lambda: mft.forward(a))

				if fft_time > mft_time:
					method = 'mft'
				else:
					method = 'fft'
	else:
		# Choose between MFT and Naive
		if input_grid.is_separated and input_grid.is_('cartesian') and output_grid.is_separated and output_grid.is_('cartesian') and input_grid.ndim in [1,2]:
			method = 'mft'
		else:
			method = 'naive'

	# Make the Fourier transform
	if method == 'fft':
		return FastFourierTransform(input_grid, q, fov)
	elif method == 'mft':
		return MatrixFourierTransform(input_grid, output_grid)
	elif method == 'naive':
		return NaiveFourierTransform(input_grid, output_grid)

class RegularCoords(CoordsBase):
	'''A list of points that have a regular spacing in all dimensions.

	Parameters
	----------
	delta : array_like
		The spacing between the points.
	dims : array_like
		The number of points along each dimension.
	zero : array_like
		The coordinates for the first point.

	Attributes
	----------
	delta
		The spacing between the points.
	dims
		The number of points along each dimension.
	zero
		The coordinates for the first point.
	'''
	def __init__(self, delta, dims, zero=None):
		if np.isscalar(dims):
			self.dims = np.array([dims]).astype('int')
		else:
			self.dims = np.array(dims).astype('int')

		if np.isscalar(delta):
			self.delta = np.array([delta]*len(self.dims))
		else:
			self.delta = np.array(delta)

		if zero is None:
			self.zero = np.zeros(len(self.dims))
		elif np.isscalar(zero):
			self.zero = np.array([zero]*len(self.dims))
		else:
			self.zero = np.array(zero)

	@property
	def separated_coords(self):
		'''A tuple of a list of the values for each dimension.

		The actual points are the iterated tensor product of this tuple.
		'''
		return [np.arange(n) * delta + zero for delta, n, zero in zip(self.delta, self.dims, self.zero)]

	@property
	def regular_coords(self):
		'''The tuple `(delta, dims, zero)` of the regularly-spaced coordinates.
		'''
		return self.delta, self.dims, self.zero

	@property
	def size(self):
		'''The number of points.
		'''
		return np.prod(self.dims)

	def __len__(self):
		'''The number of dimensions.
		'''
		return len(self.dims)

	@property
	def shape(self):
		'''The shape of an ``numpy.ndarray`` with the right dimensions.
		'''
		return self.dims[::-1]

	def __getitem__(self, i):
		'''The `i`-th point for these coordinates.
		'''
		return np.meshgrid(*self.separated_coords)[i].ravel()

	def __iadd__(self, b):
		'''Add `b` to the coordinates separately in-place.
		'''
		self.zero += b
		return self

	def __imul__(self, f):
		'''Multiply each coordinate with `f` separately in-place.
		'''
		self.delta *= f
		self.zero *= f
		return self

	def reverse(self):
		'''Reverse the ordering of points in-place.
		'''
		maximum = self.zero + self.delta * (self.dims - 1)
		self.delta = -self.delta
		self.zero = maximum
		return self

def make_focal_grid(pupil_grid, q=1, num_airy=None, focal_length=1, wavelength=1):

	f_lambda = focal_length * wavelength
	if num_airy is None:
		fov = 1
	else:
		fov = (num_airy * np.ones(pupil_grid.ndim, dtype='float')) / (pupil_grid.shape / 2)

	if np.max(fov) > 1:
		import warnings
		warnings.warn('Focal grid is larger than the maximum allowed angle (fov=%.03f). You may see wrapping when doing propagations.' % np.max(fov), stacklevel=2)

	uv = make_fft_grid(pupil_grid, q, fov)
	focal_grid = uv.scaled(f_lambda / (2*np.pi))

	return focal_grid

def circular_aperture(diameter, center=None):
	'''Makes a Field generator for a circular aperture.

	Parameters
	----------
	diameter : scalar
		The diameter of the aperture.
	center : array_like
		The center of the aperture

	Returns
	-------
	Field generator
		This function can be evaluated on a grid to get a Field.
	'''
	if center is None:
		shift = np.zeros(2)
	else:
		shift = center * np.ones(2)

	def func(grid):
		if grid.is_('cartesian'):
			x, y = grid.coords
			f = ((x-shift[0])**2 + (y-shift[1])**2) <= (diameter / 2)**2
		else:
			f = grid.r <= (diameter / 2)

		return Field(f.astype('float'), grid)

	return func

class Wavefront(object):
	def __init__(self, electric_field, wavelength=1):
		self.electric_field = electric_field
		self.wavelength = wavelength

	def copy(self):
		return copy.deepcopy(self)

	@property
	def electric_field(self):
		return self._electric_field

	@electric_field.setter
	def electric_field(self, U):
		if hasattr(U, 'grid'):
			self._electric_field = U.astype('complex')
		else:
			if len(U) == 2:
				self._electric_field = Field(U[0].astype('complex'), U[1])
			else:
				raise ValueError("Electric field requires an accompanying grid.")

	@property
	def wavenumber(self):
		return 2*np.pi / self.wavelength

	@wavenumber.setter
	def wavenumber(self, wavenumber):
		self.wavelength = 2*np.pi / wavenumber

	@property
	def grid(self):
		return self.electric_field.grid

	@property
	def intensity(self):
		return np.abs(self.electric_field)**2

	@property
	def amplitude(self):
		return np.abs(self.electric_field)

	@property
	def phase(self):
		phase = np.angle(self.electric_field)
		return Field(phase, self.electric_field.grid)

	@property
	def real(self):
		return np.real(self.electric_field)

	@property
	def imag(self):
		return np.imag(self.electric_field)

	@property
	def power(self):
		return self.intensity * self.grid.weights

	@property
	def total_power(self):
		return np.sum(self.power)

	@total_power.setter
	def total_power(self, p):
		self.electric_field *= np.sqrt(p / self.total_power)

def aberrate(aperture, N, input_grid, f=1):
    wf = Wavefront(aperture(input_grid))
    return aberrate_wf(N, wf, f)

def aberrate_wf(N, wf, f=1):
    shaped_field = wf.electric_field
    shaped_field.shape = (N, N)
    aslist = np.asarray(shaped_field).tolist()
    for rownum, row in enumerate(aslist):
        for colnum, el in enumerate(row):
            aslist[rownum][colnum] = el * np.sin(colnum * 2 * f * np.pi / N)
    wf.electric_field = Field(np.asarray(aslist).ravel(), wf.electric_field.grid)
    return wf

def pyramid_surface(refractive_index, separation, wavelength_0):
	def func(grid):
		surf = -separation / (refractive_index(wavelength_0) - 1) * (np.abs(grid.x) + np.abs(grid.y))
		return SurfaceApodizer(Field(surf, grid), refractive_index)
	return func

class PyramidWavefrontSensorOptics(object):
	def __init__(self, pupil_grid, wavelength_0=1, pupil_separation=1.5, pupil_diameter=None, num_pupil_pixels=32, q=4, refractive_index=lambda x : 1.5, num_airy=None):
		if pupil_diameter is None:
			pupil_diameter = pupil_grid.x.ptp()

		# Make mask
		sep = 0.5 * pupil_separation * pupil_diameter

		# Multiply by 2 because we want to have two pupils next to each other
		output_grid_size = (pupil_separation + 1) * pupil_diameter
		output_grid_pixels = np.ceil(num_pupil_pixels * (pupil_separation + 1))

		# Need at least two times over sampling in the focal plane because we want to separate two pupils completely
		if q < 2 * pupil_separation:
			q = 2 * pupil_separation

		# Create the intermediate and final grids
		self.output_grid = make_pupil_grid(output_grid_pixels, output_grid_size)
		self.focal_grid = make_focal_grid(pupil_grid, q=q, num_airy=num_airy, wavelength=wavelength_0)

		# Make all the optical elements
		self.pupil_to_focal = FraunhoferPropagator(pupil_grid, self.focal_grid, wavelength_0=wavelength_0)
		self.pyramid = pyramid_surface(refractive_index, sep, wavelength_0)(self.focal_grid)
		self.focal_to_pupil = FraunhoferPropagator(self.focal_grid, self.output_grid, wavelength_0=wavelength_0)

	def forward(self, wavefront):
		wf = self.pupil_to_focal.forward(wavefront)
		wf = self.pyramid.forward(wf)
		wf = self.focal_to_pupil(wf)
		return wf

class PyramidWavefrontSensorEstimator(object):
	def __init__(self, aperture, output_grid):
		self.measurement_grid = make_pupil_grid(output_grid.shape[0] / 2, output_grid.x.ptp() / 2)
		self.pupil_mask = aperture(self.measurement_grid)
		self.num_measurements = 2 * int(np.sum(self.pupil_mask > 0))

def make_pupil_grid(dims, diameter=1):
	diameter = (np.ones(2) * diameter).astype('float')
	dims = (np.ones(2) * dims).astype('int')

	delta = diameter / (dims - 1)
	zero = -diameter / 2

	return CartesianGrid(RegularCoords(delta, dims, zero))

N = 32
D = 9.96
aperture = circular_aperture(D)
pupil_grid = make_pupil_grid(N, D)
sps = 40 * N // 128
pupsep = 1
outgrid_size = int(np.ceil(sps * (pupsep + 1)))
D_grid = 3.6e-3
pyramid_grid = make_pupil_grid(N, D_grid)
nonzero_size = aperture(pupil_grid)[aperture(pupil_grid) > 0].size

def get_sub_images(electric_field):
    pyramid_grid = make_pupil_grid(N, D_grid)
    images = Field(np.asarray(electric_field).ravel(), pyramid_grid)
    pysize = int(np.sqrt(images.size))
    images.shape = (pysize, pysize)
    sub_images = [images[pysize-sps-1:pysize-1, 0:sps], images[pysize-sps-1:pysize-1, pysize-sps-1:pysize-1],
                  images[0:sps, 0:sps], images[0:sps, pysize-sps-1:pysize-1]]
    subimage_grid = make_pupil_grid(sps, D_grid * sps / N)
    for count, img in enumerate(sub_images):
        img = img.ravel()
        img.grid = subimage_grid
        sub_images[count] = img
    return sub_images

def pyramid_prop(wf):
    # Given a wavefront, returns the result of a pyramid propagation and splitting into sub-images,
    # as a list of hcipy Field objects.
    keck_pyramid = PyramidWavefrontSensorOptics(pupil_grid, pupil_separation=pupsep, num_pupil_pixels=sps)
    return get_sub_images(keck_pyramid.forward(wf).electric_field)

def estimate(images_list):
    EstimatorObject = PyramidWavefrontSensorEstimator(aperture, make_pupil_grid(sps*2, D_grid*sps*2/N))
    I_b = images_list[0]
    I_a = images_list[1]
    I_c = images_list[2]
    I_d = images_list[3]
    norm = I_a + I_b + I_c + I_d
    I_x = (I_a + I_b - I_c - I_d) / norm
    I_y = (I_a - I_b - I_c + I_d) / norm
    pygrid = make_pupil_grid(sps)
    return Field(I_x.ravel(), pygrid), Field(I_y.ravel(), pygrid)

def make_slopes(wf):
    x, y = estimate(pyramid_prop(wf))
    return np.concatenate((x, y))

def plot_on_aperture(aperture, field):
    project_onto = Wavefront(aperture(pupil_grid)).electric_field
    project_onto.shape = (N, N)

    count, i, j = 0, 0, 0
    while count < nonzero_size:
        if np.real(project_onto[i][j]) > 0:
            project_onto[i][j] = field[count]
            count += 1
        j += 1
        if j == N - 1:
            j = 0
            i += 1
    return project_onto.ravel() * aperture(pupil_grid)

def least_inv(A):
    # given a matrix A such that Ax = b, makes a least-squares matrix Y such that
    # x^ = Yb.
    return np.linalg.inv(A.T.dot(A)).dot(A.T)

def pupil_sin_phase(pupil, wavsx=1, wavsy=0, amplitude=0.1):
    size=int(np.sqrt(pupil.size))
    x=np.arange(size)
    y=np.arange(size)
    sin = np.zeros((size,size))

    if wavsx==0 and wavsy==0:
        return pupil
    elif wavsy==0:
        yfreq=0
        xfreq = 2*np.pi/((size/wavsx))
    elif wavsx==0:
        xfreq=0
        yfreq = 2*np.pi/((size/wavsy))
    else:
        xfreq = 2*np.pi/((size/wavsx))
        yfreq = 2*np.pi/((size/wavsy))

    for i in range(len(x)):
        for j in range(len(y)):
            sin[i,j] = amplitude*np.sin(xfreq*i+yfreq*j)

    return pupil*np.exp(complex(0,1)*sin).ravel()

wf = Wavefront(aperture(pupil_grid))
b = N
aberration_mode_basis = []
indices = []
print("Starting the aberration basis. Get ready for this to run for HOURS.")
for i in np.arange(0, b, 1):
    print(i)
    for j in np.arange(0, b, 1):
        aberration_mode_basis.append(pupil_sin_phase(wf.electric_field, i+1, j+1))
        indices.append([i, j])

#print("Making the pyramid basis...")
#pyramid_basis = np.asarray([make_slopes(Wavefront(x)) for x in aberration_mode_basis])
aberration_mode_basis = np.asarray([x[aperture(pupil_grid) > 0] for x in aberration_mode_basis])
print(aberration_mode_basis.shape)

print("Running Gram-Schmidt...")
orthogonalized, orth_indices = [], []
for index, x in enumerate(aberration_mode_basis):
    y = x
    for v in orthogonalized:
        y -= (x.dot(v)/v.dot(v))*v
    if abs(y.dot(y).real) > 1e-5:
        orthogonalized.append(y)
		print(len(orthogonalized))
        orth_indices.append(indices[index])

print(len(orth_indices))

def make_best_sine_approximation(wf):
    S = np.asarray(orthogonalized).T
    inversion = np.linalg.inv(S.T.dot(S))
    least_square = S.dot(inversion).dot(S.T)
    return Wavefront(Field(least_square.dot(wf.electric_field[aperture(pupil_grid) > 0]), wf.grid))

print("Testing the created method...")
original_electric = Field(pupil_sin_phase(wf.electric_field, 14, 63), wf.grid)
original = Wavefront(original_electric)
as_basis_electric = Field(plot_on_aperture(aperture, make_best_sine_approximation(original).electric_field), wf.grid)
as_basis_c = Wavefront(as_basis_electric)
as_basis_c.phase.tofile('eighteight.dat')
print("Done.")

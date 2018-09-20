#exec(open("./rotationUtility-03.py").read())
#
# Rotates vectors based on gravity, and then based on the north
#
# v is our vector
# u is the corrected vector
#
# g is our gravitational vector
# m is our magnetic field vector
# m_corr is the m-vector corrected by g-vector (to account for gravity)
#
# G is the matrix that rotates vectors based ono gravity
#	(primed coordinates are Earth coordinates)
#
#	gy = 9.8 cos( \theta )
#	gx = 9.8 sin( \theta ) sin( \phi )
#	gz = 9.8 sin( \theta ) cos( \phi )
#
#	\theta is angle between y' (Earth) and y (phone)
#	\phi is the rotation of the z (phone) axis, measured so that at \phi=0, z is north (z')
#
#
# R is the matrix that rotates our coordinates to align z with the north (z')
#
#	\psi is the angle that determines the north (z'), measured between z and z'
#
#
# M is the matrix that combines the two rotations into one: M=R*G
#
#
#


import numpy as np

def grav_rotation(g,v):
	#angles = calc_angles(g)
	angles = calc_angles2(g)
	#print "theta= ", angles[0]*180/3.14
	#print "phi= ", angles[1]*180/3.14
	#print()
	G = rotation_matrix_from_gravity( *angles )
	u = G.dot(v)
	#if u[1]>0:
	#    print("OMG, gravity is pointing up!")
	#if u[1]<0:
	#    print("gravity is pointing down")
	u = [ u[0], u[1], u[2] ]
	return u


def calc_rotation_matrix(g,m):
	# Constructing the M matrix (full roration using gravity and north infromation)
	#angles = calc_angles(g)
	angles = calc_angles2(g)
	G = rotation_matrix_from_gravity( *angles )
	#G = rotation_matrix_from_gravity_alt( *angles )
	print('\nGravitational rotation matrix =')
	print G
	m_corr = transform_normalize_magnetic_field( m, G )
	R = rotation_matrix_from_corrected_north( m_corr )
	#print R
	M = R.dot(G)
	return M


# obtain angles theta and phi from grav vector g
def calc_angles( g ):
	# Name the x,y,z components of the gravity vector
	#gx = g[1]
	#gy = g[2]
	#gz = g[3]
	gx = g[0]
	gy = g[1]
	gz = g[2]
	# Get the magnitude of the gravity vector
	# Should be around 9.8
	magnitude = np.sqrt( gx**2 + gy**2 + gz**2 )
	# Obtain angle theta (angle between y,y')
	# The new y should point downward
	# In our coordinate system (see above), gy = magnitude * cos( \theta )
	theta = np.arccos( gy/magnitude )
	# If theta is very small, then gx=0 and gz=0, so phi is arbitrary
	# In fact, phi is undefined for theta=0
	const_small = 0.00001  # defines zero
	if (-1)*const_small < np.sin(theta) < const_small:
		phi = 0
	else:
		# Recall (see above), gx = magnitude * sin( \theta ) sin( \phi )
		phi = np.arccos( gx/( magnitude*np.sin(theta) ) )
		# phi = np.arccos( gz/( magnitude*np.sin(theta) ) )
	return [theta, phi]
# end def calc_angles( g )

# obtain angles theta and phi from grav vector g
def calc_angles2( g ):
	#
	# Define the x,y,z components of the gravitational vector "g"
	gx = float(g[0])
	gy = float(g[1])
	gz = float(g[2])
	print 'g-vector = ', g
	#
	# Calculate the magnitude of g-vector
	# should be around 9.8
	#magnitude = -np.sqrt( gx**2 + gy**2 + gz**2 )
	magnitude = np.sqrt( gx**2 + gy**2 + gz**2 )
	#
	# Obtain angle theta (angle between y,y')
	# The new y should point downward
	# In our coordinate system (see above), gy = magnitude * cos( \theta )
	theta = np.arccos( gy/magnitude ) # positive number: 0<-->pi
	#
	# y should be negative, if y<0, then theta<90, if y>0, then theta>90  **check this**
	if gy<0 and theta>(0.5*np.pi):
		theta = np.pi - theta
	if gy>0 and theta<(0.5*np.pi):
		theta = np.pi - theta
	#
	# If theta is very small, then gx=0 and gz=0, so phi is arbitrary
	# In fact, phi is undefined for theta=0
	const_small = 0.00001 # defines zero
	if -const_small < np.sin(theta) < const_small:
		phi = 0
	else:
	# Recall (see above),
	# gx = magnitude * sin( \theta ) sin( \phi )
	# gz = magnitude * sin( \theta ) cos( \phi )
		#
		if abs(gz) > abs(gx):
		    phi = np.arccos( gz/( magnitude*np.sin(theta) ) )
		else:
		    phi = np.arcsin( gx/( magnitude*np.sin(theta) ) )
		#
		#phi = np.arccos( gz/( magnitude*np.sin(theta) ) ) # positive number: 0<-->pi
		#phi = np.arcsin( gx/( magnitude*np.sin(theta) ) )
	#
	print 'temp phi = ', phi
	print 'gx, gz = ', gx, '   ' , gz
	#
	if gx>0: # gx positive
		if gz>0: # should be in 1st quadrant
			#
			# if in 4th quadrant
			if phi<0:
				phi = (-1)*phi
			if 3*np.pi/2 < phi:
				phi = 2*np.pi - phi
			#
		#if gz<0: # should be in 2nd quadrant
		else: # gz<0 -- should be in 2nd quadrant
			#
			# if in 3rd quadrant
			if phi<0:
				phi = (-1)*phi
			if phi>np.pi:
				2*np.pi - phi
			#
	else: # gx negative
		if gz>0: # should be in 4th quadrant
			#
			# if in 1st quadrant
			if 0<phi<np.pi/2:
				phi = (-1)*phi
				print 'HERE'
		#if gz<0:
		else: # gz<0 -- should be in 3rd quadrant
			#
			# if in 2nd quadrant
			if np.pi/2<phi<np.pi:
				phi = 2*np.pi - phi
			#
		#
	#
	print '[theta,phi] = ',[theta, phi]
	#	
	return [theta, phi]
# end def calc_angles2( g )



def rotation_matrix_from_gravity( theta, phi ):
	# Constructing G (rotation using gravity information)
	# Rotate x-z so that z is up
	Ga = np.array([ [np.cos(phi),0,-np.sin(phi)], [0,1,0], [np.sin(phi),0,np.cos(phi)]])
	Gb = np.array([ [1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
	Gb = np.array([ [1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
	G = Gb.dot(Ga)
	return G


def transform_normalize_magnetic_field( m, G ):
	m_corr = G.dot(m)
	m_corr_xz_one = np.sqrt( m_corr[0]**2 + m_corr[2]**2 )
	m_corr_xz_norm = m_corr / m_corr_xz_one
	return m_corr_xz_norm


def rotation_matrix_from_corrected_north( m_corr_xz_norm ):
	R = np.array([ [ m_corr_xz_norm[2], 0, -m_corr_xz_norm[0] ], [0, 1, 0], [ m_corr_xz_norm[0], 0, m_corr_xz_norm[2] ] ])
	return R


def rotate( v, g, m ):
	v_t = v[0]
	v_tag = v[4]
	v = [v[1],v[2],v[3]]
	g = [g[1],g[2],g[3]]
	m = [m[1],m[2],m[3]]
	M = calc_rotation_matrix(g,m)
	u = M.dot(v)
	u = [ v_t, u[0], u[1], u[2], v_tag]
	return u


def rotate_list(v_list_in, g_list_in, m_list_in):
    print 'len(v_list_in) = ', len(v_list_in)
    v_list_out = []
    for i in range(len(v_list_in)):
        print "currently at index:",i
        #u = rotate( v_list_in[i], g_list_in[i], m_list_in[i] )
        #print v_list_in[i], "->",u
        #v_list_out.append(u)
        print 'finish rotate at index:',i

	return v_list_out



### TESTS 

degrees_to_rads = np.pi / 180.0
test_v_1 = [0, 1.0, 2.0, 3.0 ,0]

# create a g-vector from angles
# to be used by tests
def generate_G_neg( in_theta, in_phi ):
	
	theta = degrees_to_rads * in_theta
	phi   = degrees_to_rads * in_phi
	
	# NOTICE the (-) sign in the y-component
	out_G = [0, 9.8*np.sin(theta)*np.sin(phi), -9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	
	return out_G
# end def generate_G_neg


def generate_G_pos( in_theta, in_phi ):
	
	theta = degrees_to_rads * in_theta
	phi   = degrees_to_rads * in_phi
	
	#out_G = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	out_G = [ 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ]
	
	return out_G
# end def generate_G_pos



# a general testing platform
def test_general( in_vec, in_G, in_M ):
	
	print('\nTEST--')
	rotated_vec = rotate( in_vec, in_G, in_M )
	
	return rotated_vec
# end def test_general

# g-vector
# here the gravity vector is pointing upward
#
# m-vector
# pointing in the z-direction
#
# therefore
# out_x = -in_x
# out_y = -in_y
# out_z = in_z
def test1( in_vec ):
	
	theta = degrees_to_rads * 0.0
	phi   = degrees_to_rads * 0.0
	
	#theta = 0.0
	#phi   = 0.0	
	#g = generate_G_pos( theta, phi )
	#m = [0.0, 0.0, 1.0]
	
	g = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	m = [0, 0.0, 0.0, 1.0 ,0]
	
	# g = g[2:4]
	# m = m[2:4]
	
	print "\nTest1:"
	
	u = rotate( in_vec, g, m )
	
	print "in vec  = ", in_vec#, "\n"
	print "out vec = ", u#, "\n"
	print "should be [%.d,%.d,%.d]"%((-1)*in_vec[1],(-1)*in_vec[2],in_vec[3])
# end def test1( v )
#test1( test_v_1 )

# g-vector
# gravity is pointing up
#
# m-vector
# magnetic field is in the x-direction
#
# therefore
# out_x = in_z
# out_y = -in_y
# out_z = in_x
def test2( in_vec ):
	
	theta = degrees_to_rads * 0.0
	phi   = degrees_to_rads * 0.0

	g = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	m = [0, 1.0, 0.0, 0.0 ,0]
	
	# g = g[2:4]
	# m = m[2:4]
	
	print "\nTest2:"
	
	u = rotate( in_vec, g, m )
	
	print "x is north. Rotate x and z:"
	print "in v = ", in_vec#, "\n"
	print "out v= ", u#, "\n"
	print "should be [%.d,%.d,%.d]"%( in_vec[3], (-1)*in_vec[2], in_vec[1] )
# end def test2( v )
#test2( test_v_1 )

# g-vector
# gravity is pointing upward
#
# m-vector
# magnetic field is pointing opposite the z-direction
#
# therefore
# out_x = in_x
# out_y = -in_y
# out_z = -in_z
def test3( in_vec ):
	
	theta = degrees_to_rads * 0.0
	phi   = degrees_to_rads * 0.0
	
	g = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	m = [0, 0.0, 0.0, -1.0 ,0]
	
	# g = g[2:4]
	# m = m[2:4]
	
	print "\nTest3:"
	
	u = rotate( in_vec, g, m )

	print "-z is north. Rotate x and z:"
	print "in_v = ", in_vec#, "\n"
	print "out_v= ", u#, "\n"
	print "should be [%.d,%.d,%.d]"%(in_vec[1],(-1)*in_vec[2],(-1)*in_vec[3])
# end def test3( v )
#test3( test_v_1 )


# g-vector
# pointing upward
#
# m-vector
# pointing in the negative x-direction
#
# therefore
# out_x = -in_z
# out_y = -in_y
# out_z = -in_x
def test4( in_vec ):
	
	theta = degrees_to_rads * 0.0
	phi   = degrees_to_rads * 0.0
	
	g = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	m = [0, -1.0, 0.0, 0.0 ,0]
	
	# g = g[2:4]
	# m = m[2:4]
	
	print "\nTest4:"
	
	u = rotate( in_vec, g, m )
	
	print "-x is north. Rotate x and z:"
	print "in v = ", in_vec#, "\n"
	print "out v= ", u#, "\n"
	print "should be [%.d,%.d,%.d]"%((-1)*in_vec[3],(-1)*in_vec[2],(-1)*in_vec[1])
# end def test4( v )
#test4( test_v_1 )


# g-vector
# pointing in the negative z-direction
#
# m-vector
# pointing in the x-direction
#
# therefore
# out_x = in_y
# out_y = in_z
# out_z = in_x
def test5( in_vec ):
	
	theta = degrees_to_rads * 90.0
	phi   = degrees_to_rads * 180.0
	
	g = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	m = [0, 1.0, 0.0, 0.0 ,0]
	
	# g = g[2:4]
	# m = m[2:4]
	
	print "\nTest5:"
	
	u = rotate( in_vec, g, m )
	
	print "z is up. Rotate about x axis:"
	print "in v = ", in_vec#, "\n"
	print "out v= ", u#, "\n"
	print "should be [%.d,%.d,%.d]"%(in_vec[2],in_vec[3],in_vec[1])
# end def test5( v )
#test5(test_v_1)




# g-vector
# pointing in the z-direction
#
# m-vector
# pointing in the x-direction
#
# therefore
# out_x = -in_y
# out_y = -in_z
# out_z = in_x
def test6( in_vec ):
	
	theta = degrees_to_rads * 90.0
	phi   = degrees_to_rads * 0.0
	
	g = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	m = [0, 1.0, 0.0, 0.0 ,0]
	
	# g = g[2:4]
	# m = m[2:4]
	
	print "\nTest6:"
	
	u = rotate( in_vec, g, m )
	
	print "z is up. Rotate about x axis:"
	print "in v = ", in_vec#, "\n"
	print "out v= ", u#, "\n"
	print "should be [%.d,%.d,%.d]"%((-1)*in_vec[2],(-1)*in_vec[3],in_vec[1])
# end def test6( v )
#test6(test_v_1)



# g-vector
# pointing in the z-direction
#
# m-vector
# pointing in the y-direction
#
# therefore
# out_x = in_x
# out_y = -in_z
# out_z = in_y
def test7( in_vec ):
	
	theta = degrees_to_rads * 90.0
	phi   = degrees_to_rads * 0.0
	
	g = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	m = [0, 0.0, 1.0, 0.0 ,0]
	
	# g = g[2:4]
	# m = m[2:4]
	
	print "\nTest7:"
	
	u = rotate( in_vec, g, m )
	
	print "z is up. Rotate about x axis:"
	print "in v = ", in_vec#, "\n"
	print "out v= ", u#, "\n"
	print "should be [%.d,%.d,%.d]"%(in_vec[1],(-1)*in_vec[3],in_vec[2])
# end def test7( v )
#test7(test_v_1)



# g-vector
# pointing in the negative z-direction
#
# m-vector
# pointing in the y-direction
#
# therefore
# out_x = -in_x
# out_y = in_z
# out_z = in_y
def test8( in_vec ):
	
	theta = degrees_to_rads * 90.0
	phi   = degrees_to_rads * 180.0
	
	g = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	
	m = [0, 0.0, 1.0, 0.0 ,0]
	
	print "\nTest8:"
	
	u = rotate( in_vec, g, m )
	
	print "z is up. Rotate about x axis:"
	print "in v = ", in_vec#, "\n"
	print "out v= ", u#, "\n"
	print "should be [%.d,%.d,%.d]"%((-1)*in_vec[1],in_vec[3],in_vec[2])
# end def test8( v )
#test8(test_v_1)


# g-vector
# along x-axis
#
# m-vector
# along z-axis
#
# therefore
# out_x = in_y
# out_y = -in_x
# out_z = in_z
def test9( in_vec ):
	
	theta = degrees_to_rads * 90.0
	phi   = degrees_to_rads * 90.0
	
	g = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	m = [0, 0.0, 0.0, 1.0 ,0]
	
	print "\nTest9:"
	
	u = rotate( in_vec, g, m )
	
	print "z is up. Rotate about x axis:"
	print "in v = ", in_vec#, "\n"
	print "out v= ", u#, "\n"
	print "should be [%.d, %.d, %.d]"%(in_vec[2],(-1)*in_vec[1],in_vec[3])
# end def test9( v )
#test9( test_v_1 )



# g-vector
# along negative x-axis
#
# m-vector
# along z-axis
#
# therefore
# out_x = -in_y
# out_y = in_x
# out_z = in_z
def test10( in_vec ):
	
	theta = degrees_to_rads * 90.0
	phi   = degrees_to_rads * -90.0
	
	g = [0, 9.8*np.sin(theta)*np.sin(phi), 9.8*np.cos(theta), 9.8*np.sin(theta)*np.cos(phi) ,0]
	m = [0, 0.0, 0.0, 1.0 ,0]

	print "\nTest10:"
	
	u = rotate( in_vec, g, m )

	print "z is up. Rotate about x axis:"
	print "in v = ", in_vec#, "\n"
	print "out v= ", u#, "\n"
	print "should be [%.d, %.d, %.d]"%((-1)*in_vec[2],in_vec[1],in_vec[3])
# end def test10( v )
#test10( test_v_1 )

'''
test1( test_v_1 )
test2( test_v_1 )
test3( test_v_1 )
test4( test_v_1 )
test5( test_v_1 )
test6( test_v_1 )
test7( test_v_1 )
test8( test_v_1 )
test9( test_v_1 )
test10( test_v_1 )
'''

# g-vector
#
#
# m-vector
#
#
# therefore
# out_x = 
# out_y = 
# out_z = 
# 'y' (the second coordinate) should (in the end) point down (-1.0)
# def test8():
    # degrees_to_rads = np.pi / 180.0
    # theta = degrees_to_rads * 0.0
    # phi   = degrees_to_rads * 0.0
    # g = [ 0.0 ,-1.0 ,0.0 ]
    # v = [1.0, 2.0, 3.0]
    # u = grav_rotation( g, v )
    # print "No rotation:"
    # print "v= ", v#, "\n"
    # print "u= ", u#, "\n"
    # print "Should be [1,2,3]"

#
#exec(open("./rotationUtility-03.py").read())
#exec(open("C:/Users/Leon/Documents/science/suzie/rotationUtility-03.py").read())
#test7()

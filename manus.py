
import urllib, urllib.request
import json
import cv2

from math import cos, sin

import numpy as np
import numpy.matlib

from dateutil.parser import parse as dateparser

def parse_timestamp(response):
    timestamp = response.info().get("X-Timestamp", "")
    if timestamp != "":
        return dateparser(timestamp)
    else:
        return None

class RemoteAPIError(Exception):
    def __init__(self, message):
        super().__init__(message)

class RemoteAPI(object):

    def __init__(self, server):
        self.server = server

    def request_json(self, path, arguments = {}):
        try:
            response = urllib.request.urlopen(self.server.generate(path, arguments))
            timestamp = parse_timestamp(response)

            return json.loads(response.read().decode(encoding = response.info().get_content_charset('utf-8'))), timestamp
        except urllib.error.URLError as e:
            raise RemoteAPIError("Remote API error: {}".format(e))

    def post_json(self, path, data, arguments = {}):
        try:
            request = urllib.request.Request(self.server.generate(path, arguments))
            request.add_header('Content-Type', 'application/json')
            response = urllib.request.urlopen(request, json.dumps(data).encode('utf-8'))
            timestamp = parse_timestamp(response)

            return json.loads(response.read().decode(encoding = response.info().get_content_charset('utf-8'))), timestamp
        except urllib.error.URLError as e:
            raise RemoteAPIError("Remote API error: {}".format(e))

class Server(object):

    def __init__(self, address, port, protocol = "http"):
        self.address = address
        self.port = port
        self.protocol = protocol

    def generate(self, path, arguments = {}):
        if arguments:
            return "{}://{}:{}/{}?{}".format(self.protocol, self.address, self.port, path, urllib.parse.urlencode(arguments))
        else:
            return "{}://{}:{}/{}".format(self.protocol, self.address, self.port, path)

class Manipulator(RemoteAPI):

    def __init__(self, server, name = "manipulator"):
        super().__init__(server)
        self.name = name

        data, _ = self.request_json("api/manipulator/describe")
        self.joints = data["joints"]
		
        n = len(self.joints)
		
        self.initial_param = np.zeros((n, 4))
        self.initial_types = np.zeros((n, 5))

        for i in range(n):
            cur_joint = self.joints[i]
            p = np.array([cur_joint['a'], cur_joint['alpha'], cur_joint['d'], cur_joint['theta']])

            if cur_joint['type'] == 'TRANSLATION':
                self.initial_param[i, :] = p
                self.initial_types[i, :] = np.array([1, cur_joint['min'], cur_joint['max'], 3, 1])

            elif cur_joint['type'] == 'ROTATION':
                self.initial_param[i, :] = p
                self.initial_types[i, :] = np.array([0, cur_joint['min'], cur_joint['max'], 4, 1])

            elif cur_joint['type'] == 'GRIPPER':
                self.initial_param[i, :] = p
                self.initial_types[i, :] = np.array([2, cur_joint['min'], cur_joint['max'], 4, 1])

            elif cur_joint['type'] == 'FIXED':
                self.initial_param[i, :] = p
                self.initial_types[i, :] = np.array([3, 0, 0, 0, 0])

        for i in range(n):
            if self.initial_types[i, 4]:
                self.initial_param[i, self.initial_types[i, 3].astype(np.int) - 1] = self.state()[0][i]['position']

    def trajectory(self, goals, blocking = True):
        response, _ = self.post_json("api/manipulator/trajectory", goals, {"blocking" : blocking})
        return "result" in response and response["result"] == "ok"


	# Premakni skelepe robotskega manipulatorja z upostevanjem rotacij specificiranih v goals input argumentu
    def move(self, goals, speed = 1.0, blocking = True):
        response, _ = self.post_json("api/manipulator/move", [{"goals" : goals, "speed" : speed}], {"blocking" : blocking})
        return "result" in response and response["result"] == "ok"

    # Premakni posamezni sklep manipolatorja
    def joint(self, id, goal, speed=1.0, blocking=True):
        response, _ = self.post_json("api/manipulator/joint", {'id': id, 'goal': goal, 'speed': speed}, {"blocking": blocking})
        return "result" in response and response["result"] == "ok"

	# Odcitaj trenutne rotacije sklepov manipulatorja
    def state(self):
        state, timestamp = self.request_json("api/manipulator/state")
        return state["joints"], timestamp

	# Izracunaj matriko F (za pridobitev konca robatskega manipulatorja)
    def transformation(self, state):

        F = np.eye(4)

        for j, s in zip(self.joints, state):

            alpha = j["alpha"]
            theta = j["theta"]
            d = j["d"]
            a = j["a"]
            if j["type"] == "ROTATION":
                theta = s["position"]

            if j["type"] == "GRIPPER":
                break

            M = np.array([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
             [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
             [0, sin(alpha), cos(alpha), d], [0, 0, 0, 1]])

            F = np.matmul(F, M)

        return F


	# Pridobi X,Y,Z koordinate konca robotskega manipulatorja na podlagi trenutnih rotacij sklepov
    def position(self, state):

        F = self.transformation(state)

        P = np.matmul(F, np.array([[0], [0], [0], [1]]))

        return np.squeeze([P[0], P[1], P[2]])
		
		
    def solve(self, position, iterations=40, distance=1):
        # Get degrees of freedom (DOF) of manipulator
        n = len(self.joints)
        origin = np.array([0, 0, 0])

        # Initialize current state
        state = np.zeros(n)
        for i in range(n):
            if self.initial_types[i, 4]:
                state[i] = self.joints[i]['theta']

        # Iterate
        for k in range(iterations):
            for i in range(n):
                state = self.ik_optimize_joint(position, state, i, 600)

            parameters = np.copy(self.initial_param)

            for i in range(n):
                if self.initial_types[i, 4]:
                    parameters[i, self.initial_types[i, 3].astype(np.int) - 1] = state[i]

            # Get current position
            cur_position = self.calculate_positions(parameters, origin)

            proximity = np.sqrt(np.sum((cur_position - position) ** 2))

            if proximity < distance:
                break

        # Return state as list
        return state.tolist()


    def ik_optimize_joint(self, position, start, joint, N):
        state = np.copy(start)

        origin = np.array([0, 0, 0])

        if self.initial_types[joint, 0] > 1:
            return state

        nj = len(start)

        parameters = np.reshape(np.matlib.repmat(self.initial_param, 1, N), (nj, 4, N), order='F')
        samples = np.linspace(self.initial_types[joint, 1], self.initial_types[joint, 2], N)

        for i in range(nj):
            if self.initial_types[i, 4]:
                parameters[i, self.initial_types[i, 3].astype(np.int) - 1, :] = start[i]

        parameters[joint, self.initial_types[joint, 3].astype(np.int) - 1, :] = samples    
   
        positions = self.calculate_positions(parameters, origin)
    
        proximity = np.sqrt(np.sum((positions - np.reshape(np.matlib.repmat(position, 1, N), (3, 1, N), order='F')) ** 2, 0))
    
        best_i = np.argmin(proximity)

        state[joint] = samples[best_i]
        return state


    def calculate_positions(self, parameters, origin):
        if len(parameters.shape) < 3:
            j, a = parameters.shape
            vn = 1
            tmp = np.zeros((j,a,1))
            tmp[:,:,0] = parameters
            parameters = np.copy(tmp)
        else:
            j, a, vn = parameters.shape
        S = np.reshape(np.matlib.repmat(np.eye(4), 1, vn), (4, 4, vn), order='F')

        V = np.zeros((4, 4, vn))
        V[3, 3, :] = 1

        for i in range(j):
            V[0, 0, :] = np.cos(parameters[i, 3, :])
            V[1, 0, :] = np.sin(parameters[i, 3, :])

            V[0, 1, :] = np.multiply(-np.sin(parameters[i, 3, :]), np.cos(parameters[i, 1, :]))
            V[1, 1, :] = np.multiply(np.cos(parameters[i, 3, :]), np.cos(parameters[i, 1, :]))
            V[2, 1, :] = np.sin(parameters[i, 1, :])

            V[0, 2, :] = np.multiply(np.sin(parameters[i, 3, :]), np.sin(parameters[i, 1, :]))
            V[1, 2, :] = np.multiply(-np.cos(parameters[i, 3, :]), np.sin(parameters[i, 1, :]))
            V[2, 2, :] = np.cos(parameters[i, 1, :])

            V[0, 3, :] = np.multiply(parameters[i, 0, :], np.cos(parameters[i, 3, :]))
            V[1, 3, :] = np.multiply(parameters[i, 0, :], np.sin(parameters[i, 3, :]))
            V[2, 3, :] = parameters[i, 2, :]
	   
            S = self.multiprod(S, V)

        positions = self.multiprod(S, np.reshape(np.matlib.repmat(np.concatenate((origin, np.array([1]))), 1, vn), (4, 1, vn), order='F'))

        return positions[0:3, :, :]


    def multiprod(self, A, B):
        if len(np.shape(A)) == 2:
            return np.dot(A, B)

        return np.einsum('ijk,jlk->ilk', A, B)


class Camera(RemoteAPI):

    def __init__(self, server, name = "camera"):
        super().__init__(server)
        self.name = name

        data, _ = self.request_json("api/camera/describe")
        self.intrinsics = np.array(data["intrinsics"])
        self.distortion = np.array(data["distortion"])

    def position(self):
        data, timestamp = self.request_json("api/camera/position")

        rotation = np.array(data["rotation"])
        translation = np.array(data["translation"])

        return rotation, translation, timestamp

    def image(self):
        try:
            response = urllib.request.urlopen(self.server.generate("api/camera/image"))

            timestamp = parse_timestamp(response)

            image = np.asarray(bytearray(response.read()), dtype="uint8")
            return cv2.imdecode(image, cv2.IMREAD_COLOR), timestamp
        except urllib.error.URLError as e:
            raise RemoteAPIError("Remote API error: {}".format(e))
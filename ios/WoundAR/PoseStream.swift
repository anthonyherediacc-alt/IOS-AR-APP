import Foundation
import Network

// Main queue only. One authenticated client; at most one send in flight and 10 Hz.
// USB: iproxy forwards desktop localhost:8765 to this device's TCP port 8765.
final class PoseStream {
    var onStatus: ((String) -> Void)?
    private var listener: NWListener?
    private var client: NWConnection?
    private var authorized = false
    private var sending = false
    private var awaitingAck = false
    private var lastSent: Double = -.infinity
    private var code = ""

    func start(code: String) {
        stop(); self.code = code
        do {
            let listener = try NWListener(using: .tcp, on: 8765)
            self.listener = listener
            listener.stateUpdateHandler = { [weak self, weak listener] state in
                guard let self, let listener, self.listener === listener else { return }
                switch state {
                case .ready: self.onStatus?("Listening on TCP 8765 • waiting for Unity")
                case .failed(let error): self.stop(); self.onStatus?("Stream failed: \(error.localizedDescription)")
                default: break
                }
            }
            listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
            listener.start(queue: .main)
        } catch { onStatus?("Stream failed: \(error.localizedDescription)") }
    }
    func stop() {
        listener?.cancel(); listener = nil
        client?.cancel(); client = nil; authorized = false; sending = false; awaitingAck=false; lastSent = -.infinity
        onStatus?("Stopped")
    }
    private func accept(_ connection: NWConnection) {
        guard client == nil else { connection.cancel(); return }
        client = connection; authorized = false; sending = false; awaitingAck=false
        connection.stateUpdateHandler = { [weak self, weak connection] state in
            guard let self, let connection, self.client === connection else { return }
            switch state {
            case .failed, .cancelled: self.disconnect(connection)
            default: break
            }
        }
        connection.start(queue: .main)
        receive(connection, bytes: Data())
        DispatchQueue.main.asyncAfter(deadline: .now()+5) { [weak self, weak connection] in
            guard let self, let connection, self.client === connection, !self.authorized else { return }
            self.disconnect(connection)
        }
    }
    private func disconnect(_ connection: NWConnection) {
        guard client === connection else { return }
        client = nil; authorized = false; sending = false; awaitingAck=false; connection.cancel()
        onStatus?("Client disconnected • waiting for Unity")
    }
    private func receive(_ connection: NWConnection, bytes: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 128) { [weak self, weak connection] data, _, done, error in
            guard let self, let connection, self.client === connection else { return }
            if done || error != nil { self.disconnect(connection); return }
            var buffer = bytes; if let data { buffer.append(data) }
            if self.authorized {
                // Frame-level backpressure prevents queued old geometry on a slow link.
                if buffer == Data("ACK\n".utf8), self.awaitingAck {
                    self.awaitingAck=false; self.receive(connection,bytes:Data())
                } else if buffer.count<4, Data("ACK\n".utf8).starts(with:buffer) {
                    self.receive(connection,bytes:buffer)
                } else { self.disconnect(connection) }
                return
            }
            if let newline = buffer.firstIndex(of: 10) {
                guard String(data: buffer.prefix(upTo: newline), encoding: .utf8) == self.code,
                      newline == buffer.count-1 else { self.disconnect(connection); return }
                self.authorized = true; self.lastSent = -.infinity
                self.onStatus?("Connected • streaming pose + LiDAR samples at up to 10 Hz")
                self.receive(connection, bytes: Data())
            } else if buffer.count > 64 { self.disconnect(connection) }
            else { self.receive(connection, bytes: buffer) }
        }
    }
    func send(_ snapshot: BodySnapshot) {
        guard let client, authorized, !sending, !awaitingAck, snapshot.packet.timestamp-lastSent >= 0.1 else { return }
        do {
            let data = try LiveFrame(packet: snapshot.packet, surface: snapshot.surface, mesh: snapshot.mesh).encodedLine()
            sending = true; awaitingAck=true; lastSent = snapshot.packet.timestamp
            let sentAt=lastSent
            DispatchQueue.main.asyncAfter(deadline:.now()+3) { [weak self, weak client] in
                guard let self, let client, self.client === client, self.awaitingAck, self.lastSent == sentAt else { return }
                self.disconnect(client)
            }
            client.send(content: data, completion: .contentProcessed { [weak self, weak client] error in
                guard let self, let client, self.client === client else { return }
                self.sending = false
                if error != nil { self.disconnect(client) }
            })
        } catch { onStatus?("Frame skipped: \(error.localizedDescription)") }
    }
}

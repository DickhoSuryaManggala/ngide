require 'json'
require 'fileutils'

# Ruby Automation Script for Institutional Reporting & Data Management
class InstitutionalAutomation
  DATA_DIR = "data"
  REPORTS_DIR = "reports/institutional"

  def initialize
    FileUtils.mkdir_p(REPORTS_DIR)
  end

  # Membersihkan data lama secara otomatis (Maintenance)
  def cleanup_old_data(days = 30)
    puts "--- Ruby Automation: Cleaning up data older than #{days} days ---"
    Dir.glob("#{DATA_DIR}/*.csv").each do |file|
      if File.mtime(file) < Time.now - (days * 24 * 60 * 60)
        puts "Archiving old file: #{file}"
        FileUtils.mv(file, "#{REPORTS_DIR}/archive_#{File.basename(file)}")
      end
    end
  end

  # Menghasilkan ringkasan status sistem untuk tim IT (DevOps Standard)
  def generate_system_health_report
    health = {
      timestamp: Time.now.to_s,
      status: "HEALTHY",
      memory_usage: `free -m | grep Mem | awk '{print $3}'`.strip + " MB",
      disk_free: `df -h / | tail -1 | awk '{print $4}'`.strip,
      active_processes: `ps aux | grep python3 | wc -l`.strip.to_i
    }
    
    File.write("#{REPORTS_DIR}/system_health.json", JSON.pretty_generate(health))
    puts "--- Ruby Automation: System Health Report Generated ---"
  end
end

if __FILE__ == $0
  auto = InstitutionalAutomation.new
  auto.cleanup_old_data
  auto.generate_system_health_report
end
